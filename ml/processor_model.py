import torch
from torch import nn

from graph import build_particle_edges, build_wall_ghosts
from structured_model import (ParticleInteractionModel,project_pair_motion,aggregate_pair_impulses,integrate_state)

def build_pair_channels(state,pairs,velocity_scale,spin_scale,radii,length_floor=1e-6,tangent_cross=False, tangent_projection=False):
    i,j = pairs
    displacement = state[j,:3]-state[i,:3]
    a = displacement / torch.sqrt(displacement.square().sum(dim=1,keepdim=True)+length_floor**2)
    normal = displacement / displacement.norm(dim=1, keepdim=True).clamp_min(length_floor)
    def smooth_bound(vector):
        return vector / torch.sqrt(1.0+vector.square().sum(dim=1,keepdim=True))
    relative_surface_velocity = (state[j, 3:6] - state[i, 3:6] - torch.cross(radii[i, None] * state[i, 6:9] + radii[j, None] * state[j, 6:9],normal,dim=1))
    tangential_velocity = relative_surface_velocity - (relative_surface_velocity * normal).sum(dim=1, keepdim=True) * normal
    c = smooth_bound(tangential_velocity / velocity_scale)
    if tangent_cross:
        b = torch.cross(a, c, dim=1)
    else:
        velocity_i = state[i, 3:6] / velocity_scale
        velocity_j = state[j, 3:6] / velocity_scale
        spin_i = state[i, 6:9] / spin_scale
        spin_j = state[j, 6:9] / spin_scale
        reference = (velocity_i + velocity_j + spin_i + spin_j + torch.cross(velocity_j - velocity_i, a, dim=1) + torch.cross(spin_j - spin_i, a, dim=1))
        parallel = (reference * a).sum(dim=1, keepdim=True) * a
        b = smooth_bound(torch.cross(reference - parallel, a, dim=1))
        if tangent_projection:
            projection_floor = 0.1
            along_c = (b * c).sum(dim=1, keepdim=True)
            c_squared = c.square().sum(dim=1, keepdim=True)
            b = b - (along_c / (c_squared + projection_floor**2)) * c
    def condition(vector):
        floor = 0.1
        return vector / torch.sqrt(
            vector.square().sum(dim=1, keepdim=True) + floor**2
        )
    return torch.stack([a, condition(b), condition(c)], dim=1)

class ProcessorModel(ParticleInteractionModel):
    def __init__(self,*args,rounds=3,interaction_gate=False,early_gate=False,tangent_cross=False,tangent_projection=False,swap_aware_pairs=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.rounds = rounds
        self.interaction_gate = interaction_gate
        self.early_gate = early_gate
        self.tangent_cross = tangent_cross
        self.tangent_projection = tangent_projection
        self.swap_aware_pairs = swap_aware_pairs
        width = self.encoder.motion_encoder[-1].out_features
        if swap_aware_pairs:
            # Preserve the training RNG sequence used by the control.
            with torch.random.fork_rng(devices=[]):
                self.decoder.output = nn.Linear(width, 12)
            nn.init.zeros_(self.decoder.output.weight)
            nn.init.zeros_(self.decoder.output.bias)
        self.pair_norm = nn.LayerNorm(width)
        self.wall_norm = nn.LayerNorm(width)
    def encode_interactions(self,state,previous,pairs,geometry,encoder,normalization,memory,swap_aware=False):
        ghost_count = state.shape[0] - self.radii.numel()
        surface_radii = torch.cat([self.radii,self.radii.new_zeros(ghost_count)])
        frames = build_pair_channels(state,pairs,velocity_scale=self.motion_scale[0],spin_scale=self.motion_scale[3],radii=surface_radii,tangent_cross=self.tangent_cross or swap_aware,tangent_projection=self.tangent_projection)
        motion_i, motion_j = project_pair_motion(state, previous, pairs, frames)
        motion_i = motion_i / self.motion_scale
        motion_j = motion_j / self.motion_scale
        odd_b = None
        if swap_aware:
            # project_pair_motion already negates j's frame.
            # Under a swap, the two b slices exchange and negate.
            odd_b = motion_i[:, 1::3] + motion_j[:, 1::3]
            motion_i = torch.stack((motion_i[:, 0::3], motion_i[:, 1::3].abs(), motion_i[:, 2::3]),dim=-1,).flatten(1)
            motion_j = torch.stack((motion_j[:, 0::3], motion_j[:, 1::3].abs(), motion_j[:, 2::3]),dim=-1).flatten(1)
        embedding = encoder(motion_i, motion_j, geometry)
        memory = normalization(embedding + memory)
        return frames, memory, odd_b
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
            frames, pair_memory, odd_b = self.encode_interactions(state,history,pairs,geometry,self.encoder,self.pair_norm,pair_memory,swap_aware=self.swap_aware_pairs)
            linear, angular, points = self.decoder(pair_memory,frames,positions,pairs,odd_b=odd_b)
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
                wall_frames, updated_memory, _ = self.encode_interactions(combined,combined_history,wall_pairs,wall_geometry,self.wall_encoder,self.wall_norm,wall_memory[wall_keys])
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
            state = integrate_state(state,net_linear,net_spin,self.masses,self.inertias,internal_dt,reference_dt=getattr(self,"integration_reference_dt", None),contact_position_fraction=getattr(self, "contact_position_fraction", 1.0))
        return state



