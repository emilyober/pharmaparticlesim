import torch
from torch import nn
from graph import build_particle_edges, build_wall_ghosts

def aggregate_pair_impulses(positions,pairs,linear_impulses,angular_impulses,application_points):
    i,j = pairs
    impulse_i = linear_impulses
    impulse_j = -linear_impulses
    spin_i = angular_impulses - torch.cross(positions[i]-application_points,impulse_i,dim=1)
    spin_j = -angular_impulses - torch.cross(positions[j]-application_points,impulse_j,dim=1)
    net_linear = torch.zeros_like(positions)
    net_spin = torch.zeros_like(positions)
    net_linear.index_add_(0,i,impulse_i)
    net_linear.index_add_(0,j,impulse_j)
    net_spin.index_add_(0,i,spin_i)
    net_spin.index_add_(0,j,spin_j)
    return net_linear,net_spin

def integrate_state(state,net_linear,net_spin,masses,inertias,step_dt):
    gravity = state.new_tensor([0.0, 0.0, -9.81])
    positions = state[:, :3]
    velocities = state[:, 3:6]
    spins = state[:, 6:9]
    new_velocities = (velocities+gravity*step_dt+net_linear/masses[:,None])
    new_spins = (spins+net_spin/inertias[:,None])
    new_positions = positions+new_velocities*step_dt
    return torch.cat([new_positions,new_velocities,new_spins],dim=1)

def normalize_or_zero(vector,eps=1e-8):
    magnitude = vector.norm(dim=-1,keepdim=True)
    return vector/magnitude.clamp_min(eps)

def build_pair_frame(state,pairs):
    i,j = pairs
    displacement = state[j,:3]-state[i,:3]
    a = normalize_or_zero(displacement)
    velocity_i = state[i,3:6]
    velocity_j = state[j,3:6]
    spin_i = state[i,6:9]
    spin_j = state[j,6:9]
    reference = (normalize_or_zero(velocity_i+velocity_j)+normalize_or_zero(spin_i+spin_j)+normalize_or_zero(torch.cross(velocity_j-velocity_i,displacement,dim=-1))+normalize_or_zero(torch.cross(spin_j-spin_i,displacement,dim=-1)))
    parallel = (reference*a).sum(dim=-1,keepdim=True)*a
    perpendicular = reference-parallel
    b = normalize_or_zero(torch.cross(perpendicular,a,dim=-1))
    c = normalize_or_zero(torch.cross(parallel,b,dim=-1))
    return torch.stack([a,b,c],dim=1)

def project_pair_motion(state,previous_state,pairs,frames):
    i,j = pairs
    motion = torch.stack([state[:,3:6],state[:,6:9],previous_state[:,3:6],previous_state[:,6:9]], dim=1)
    projected_i = torch.einsum("pvc,pbc->pvb", motion[i],frames)
    projected_j = torch.einsum("pvc,pbc->pvb", motion[j], -frames)
    return projected_i.flatten(1),projected_j.flatten(1)


class PairEncoder(nn.Module):
    def __init__(self,hidden_size=64):
        super().__init__()
        self.motion_encoder = nn.Sequential(
            nn.Linear(12,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(2,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size)
        )
        self.interaction_encoder = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size)
        )
    def forward(self,motion_i,motion_j,geometry):
        combined = (self.motion_encoder(motion_i)+self.motion_encoder(motion_j)+self.geometry_encoder(geometry))
        return self.interaction_encoder(combined)

class PairDecoder(nn.Module):
    def __init__(self,linear_impulse_scale,angular_impulse_scale,hidden_size=64):
        super().__init__()
        self.output = nn.Linear(hidden_size,6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("linear_scale",torch.tensor(float(linear_impulse_scale)))
        self.register_buffer("angular_scale",torch.tensor(float(angular_impulse_scale)))
    def forward(self,embeddings,frames,positions,pairs):
        coefficients = self.output(embeddings)
        linear_impulses = torch.einsum("pb,pbc->pc",coefficients[:,:3],frames)*self.linear_scale
        angular_impulses = torch.einsum("pb,pbc->pc",coefficients[:,3:],frames)*self.angular_scale
        i,j = pairs
        application_points = 0.5*(positions[i]+positions[j])
        return linear_impulses,angular_impulses,application_points


class ParticleInteractionModel(nn.Module):
    def __init__(self,masses,radii,velocity_scale,spin_scale,step_dt=50e-6,cutoff=0.0025,length_scale=0.002,hidden_size=64):
        super().__init__()
        masses = torch.as_tensor(masses,dtype=torch.float32).clone()
        radii = torch.as_tensor(radii,dtype=torch.float32).clone()
        inertias = 0.4*masses*radii.square()
        if masses.ndim != 1 or masses.shape != radii.shape:
            raise ValueError("masses and radii must have matching shapes")
        self.register_buffer("masses",masses)
        self.register_buffer("radii",radii)
        self.register_buffer("inertias",inertias)
        self.register_buffer("motion_scale",torch.tensor([velocity_scale,spin_scale,velocity_scale,spin_scale],dtype=torch.float32).repeat_interleave(3))
        self.register_buffer("length_scale",torch.tensor(float(length_scale)))
        self.step_dt = step_dt
        self.cutoff = cutoff
        self.encoder = PairEncoder(hidden_size)
        self.decoder = PairDecoder(linear_impulse_scale=masses.mean().item()*velocity_scale,angular_impulse_scale=inertias.mean().item()*spin_scale,hidden_size=hidden_size)
        self.wall_encoder = PairEncoder(hidden_size)
        self.wall_decoder = PairDecoder(linear_impulse_scale=masses.mean().item()*velocity_scale,angular_impulse_scale=inertias.mean().item()*spin_scale,hidden_size=hidden_size)

    def wall_impulses(self,state,previous_state):
        ghosts = build_wall_ghosts(state,previous_state,self.radii)
        num_particles = state.shape[0]
        num_ghosts = ghosts["owners"].numel()
        if num_ghosts == 0:
            return (torch.zeros_like(state[:, :3]), torch.zeros_like(state[:, :3]))
        combined_state = torch.cat([state,ghosts["current"]],dim=0)
        combined_previous = torch.cat([previous_state,ghosts["previous"]],dim=0)
        ghost_indices = torch.arange(num_particles,num_particles+num_ghosts,device=state.device)
        pairs = torch.stack([ghosts["owners"],ghost_indices])
        frames = build_pair_frame(combined_state,pairs)
        motion_i,motion_j = project_pair_motion(combined_state,combined_previous,pairs,frames)
        i,j = pairs
        distance = (combined_state[j, :3] - combined_state[i, :3]).norm(dim=1)
        geometry = torch.stack([distance / self.length_scale,-ghosts["gaps"] / self.length_scale], dim=1)
        embeddings = self.wall_encoder(motion_i / self.motion_scale, motion_j / self.motion_scale, geometry)
        linear, angular, points = self.wall_decoder(embeddings,frames,combined_state[:, :3],pairs)
        net_linear, net_spin = aggregate_pair_impulses(combined_state[:, :3],pairs,linear,angular,points)
        return (net_linear[:num_particles],net_spin[:num_particles])

    def forward(self,state,previous_state):
        positions = state[:,:3]
        edge_index = build_particle_edges(positions,self.cutoff)
        pairs = edge_index[:,:edge_index.shape[1]//2]
        frames = build_pair_frame(state,pairs)
        motion_i,motion_j = project_pair_motion(state,previous_state,pairs,frames)
        motion_i = motion_i/self.motion_scale
        motion_j = motion_j / self.motion_scale
        i, j = pairs
        distance = (positions[j]-positions[i]).norm(dim=1)
        overlap = self.radii[i]+self.radii[j]-distance
        geometry = torch.stack([distance/self.length_scale,overlap/self.length_scale],dim=1)
        embeddings = self.encoder(motion_i,motion_j,geometry)
        linear,angular,points = self.decoder(embeddings,frames,positions,pairs)
        net_linear,net_spin = aggregate_pair_impulses(positions,pairs,linear,angular,points)
        wall_linear, wall_spin = self.wall_impulses(state, previous_state)
        net_linear = net_linear + wall_linear
        net_spin = net_spin + wall_spin
        return integrate_state(state,net_linear,net_spin,self.masses,self.inertias,self.step_dt)



if __name__ == "__main__":
    positions = torch.tensor([[-0.001, 0.0, 0.0], [0.001, 0.0, 0.0]],dtype=torch.float64)
    pairs = torch.tensor([[0], [1]])
    linear_impulses = torch.tensor([[0.01, 0.02, 0.0]], dtype=torch.float64)
    angular_impulses = torch.tensor([[0.0, 0.0, 0.003]], dtype=torch.float64)
    application_points = torch.tensor([[0.0002, 0.0003, 0.0]], dtype=torch.float64)
    net_linear, net_spin = aggregate_pair_impulses(positions,pairs,linear_impulses,angular_impulses,application_points)
    total_linear = net_linear.sum(dim=0)
    total_angular = (torch.cross(positions, net_linear, dim=1) + net_spin).sum(dim=0)

    print("Linear balance error:", total_linear.abs().max().item())
    print("Angular balance error:", total_angular.abs().max().item())