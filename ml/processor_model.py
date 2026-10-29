import torch
from torch import nn

from graph import build_particle_edges, build_wall_ghosts
from structured_model import (ParticleInteractionModel,build_pair_frame,project_pair_motion,aggregate_pair_impulses,integrate_state)

def build_pair_channels(state,pairs,velocity_scale,spin_scale,radii,length_floor=1e-6):
    i,j = pairs
    displacement = state[j,:3]-state[i,:3]
    a = displacement / torch.sqrt(displacement.square().sum(dim=1,keepdim=True)+length_floor**2)
    velocity_i = state[i,3:6] / velocity_scale
    velocity_j = state[j,3:6] / velocity_scale
    spin_i = state[i, 6:9] / spin_scale
    spin_j = state[j, 6:9] /spin_scale
    reference = (velocity_i+velocity_j+spin_i+spin_j+torch.cross(velocity_j-velocity_i,a,dim=1)+torch.cross(spin_j-spin_i,a,dim=1))
    parallel = (reference*a).sum(dim=1,keepdim=True)*a
    perpendicular = reference - parallel
    def smooth_bound(vector):
        return vector / torch.sqrt(1.0+vector.square().sum(dim=1,keepdim=True))
    b = smooth_bound(torch.cross(perpendicular,a,dim=1))
    normal = displacement / displacement.norm(dim=1,keepdim=True).clamp_min(length_floor)
    relative_surface_velocity = (state[j,3:6]-state[i,3:6]-torch.cross(radii[i,None]*state[i,6:9]+radii[j,None]*state[j,6:9],normal,dim=1))
    tangential_velocity = relative_surface_velocity - (relative_surface_velocity*normal).sum(dim=1,keepdim=True)*normal
    c = smooth_bound(tangential_velocity/velocity_scale)
    def condition(vector):
        floor = 0.1
        return vector / torch.sqrt(
            vector.square().sum(dim=1, keepdim=True) + floor**2
        )
    return torch.stack([a, condition(b), condition(c)], dim=1)

class ProcessorModel(ParticleInteractionModel):
    def __init__(self,*args,rounds=3,interaction_gate=False,early_gate=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.rounds = rounds
        self.interaction_gate = interaction_gate
        self.early_gate = early_gate
        width = self.encoder.motion_encoder[-1].out_features
        self.pair_norm = nn.LayerNorm(width)
        self.wall_norm = nn.LayerNorm(width)
    def encode_interactions(self,state,previous,pairs,geometry,encoder,normalization,memory):
        ghost_count = state.shape[0] - self.radii.numel()
        surface_radii = torch.cat([self.radii,self.radii.new_zeros(ghost_count)])
        frames = build_pair_channels(state,pairs,velocity_scale=self.motion_scale[0],spin_scale=self.motion_scale[3],radii=surface_radii)
        motion_i, motion_j = project_pair_motion(state,previous,pairs,frames)
        embedding = encoder(motion_i/self.motion_scale,motion_j/self.motion_scale,geometry)
        memory = normalization(embedding+memory)
        return frames, memory
    def forward(self, state, previous_state):
        num_particles = state.shape[0]
        width = self.pair_norm.normalized_shape[0]
        internal_dt = self.step_dt/self.rounds
        use_early_gate = (self.early_gate and self.interaction_gate and self.rounds == 1 and not self.training and not torch.is_grad_enabled())
        edges = build_particle_edges(state[:,:3],self.cutoff)
        pairs = edges[:,:edges.shape[1]//2]
        if use_early_gate:
            pi, pj = pairs
            gap = -(self.radii[pi] + self.radii[pj] - (state[pj, :3] - state[pi, :3]).norm(dim=1))
            reachable = ((state[pj, 3:6] - state[pi, 3:6]).norm(dim=1) * internal_dt + 1e-6)
            pairs = pairs[:, gap <= reachable]
        i,j = pairs
        pair_memory = state.new_zeros((pairs.shape[1],width))
        wall_memory = state.new_zeros((3*num_particles,width))
        history = previous_state
        for round in range(self.rounds):
            positions = state[:,:3]
            distance = (positions[j]-positions[i]).norm(dim=1)
            overlap = (self.radii[i]+self.radii[j]-distance)
            geometry = torch.stack([distance/self.length_scale,overlap/self.length_scale],dim=1)
            frames, pair_memory = self.encode_interactions(state,history,pairs,geometry,self.encoder,self.pair_norm,pair_memory)
            linear, angular, points = self.decoder(pair_memory,frames,positions,pairs)
            if self.interaction_gate:
                relative_speed = (state[j, 3:6] - state[i, 3:6]).norm(dim=1)
                reachable = relative_speed * internal_dt + 1e-6
                gate = (-overlap <= reachable).to(state.dtype)[:, None]
                linear = linear * gate
                angular = angular * gate
            net_linear, net_spin = aggregate_pair_impulses(positions,pairs,linear/self.rounds,angular/self.rounds,points)
            ghosts = build_wall_ghosts(state,history,self.radii)
            if use_early_gate:
                reachable = (state[ghosts["owners"], 3:6].norm(dim=1) * internal_dt + 0.5 * 9.81 * internal_dt**2 + 1e-6)
                keep = ghosts["gaps"] <= reachable
                ghosts = {
                    key: value[keep]
                    for key, value in ghosts.items()
                }
            owners = ghosts["owners"]
            num_ghosts = owners.numel()
            if num_ghosts:
                combined = torch.cat([state,ghosts["current"]],dim=0)
                combined_history = torch.cat([history,ghosts["previous"]],dim=0)
                ghost_indices = torch.arange(num_particles,num_particles+num_ghosts,device=state.device)
                wall_pairs = torch.stack([owners,ghost_indices])
                wall_keys = (ghosts["wall_ids"]*num_particles+owners)
                wall_distance = (combined[ghost_indices,:3]-combined[owners,:3]).norm(dim=1)
                wall_geometry = torch.stack([wall_distance/self.length_scale,-ghosts["gaps"]/self.length_scale],dim=1)
                wall_frames, updated_memory = self.encode_interactions(combined,combined_history,wall_pairs,wall_geometry,self.wall_encoder,self.wall_norm,wall_memory[wall_keys])
                wall_memory = torch.zeros_like(wall_memory).index_copy(0,wall_keys,updated_memory)
                wall_linear, wall_angular, wall_points = (self.wall_decoder(updated_memory,wall_frames,combined[:,:3],wall_pairs))
                if self.interaction_gate:
                    speed = state[owners, 3:6].norm(dim=1)
                    reachable = (
                        speed * internal_dt
                        + 0.5 * 9.81 * internal_dt**2
                        + 1e-6
                    )

                    gate = (
                        ghosts["gaps"] <= reachable
                    ).to(state.dtype)[:, None]

                    wall_linear = wall_linear * gate
                    wall_angular = wall_angular * gate
                wall_net_linear,wall_net_spin = (aggregate_pair_impulses(combined[:,:3],wall_pairs,wall_linear/self.rounds,wall_angular/self.rounds,wall_points))
                net_linear = (net_linear+wall_net_linear[:num_particles])
                net_spin = (net_spin+wall_net_spin[:num_particles])
            else:
                wall_memory = torch.zeros_like(wall_memory)
            state = integrate_state(state,net_linear,net_spin,self.masses,self.inertias,internal_dt)
        return state



