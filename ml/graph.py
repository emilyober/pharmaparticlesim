import torch
from scipy.spatial import KDTree

def build_particle_edges(positions,cutoff=0.0025):
    pairs = KDTree(positions.detach().cpu().numpy()).query_pairs(cutoff,output_type="ndarray")
    pairs = torch.as_tensor(pairs,dtype=torch.long,device=positions.device)
    return torch.cat([pairs.T,pairs.T.flip(0)],dim=1)

def particle_edge_geometry(state,radii,edge_index):
    sender, receiver = edge_index
    displacement = state[receiver,:3]-state[sender,:3]
    distance = displacement.norm(dim=1,keepdim=True)
    normal = displacement / distance.clamp_min(1e-8)
    relative_velocity = (state[receiver,3:6]-state[sender,3:6])
    sender_arm = radii[sender,None]*normal
    receiver_arm = -radii[receiver,None]*normal
    sender_surface_velocity = (state[sender,3:6]+torch.cross(state[sender,6:9], sender_arm, dim=1))
    receiver_surface_velocity = (state[receiver,3:6]+torch.cross(state[receiver,6:9],receiver_arm,dim=1))
    relative_surface_velocity = (receiver_surface_velocity - sender_surface_velocity)
    normal_speed = (relative_surface_velocity*normal).sum(dim=1,keepdim=True)
    tangential_velocity = (relative_surface_velocity - normal_speed*normal)
    overlap = radii[sender] + radii[receiver] - distance.squeeze(1)
    return {
        "displacement": displacement,
        "distance": distance,
        "normal": normal,
        "relative_velocity": relative_velocity,
        "overlap": overlap,
        "relative_surface_velocity": relative_surface_velocity,
        "normal_speed": normal_speed,
        "tangential_velocity": tangential_velocity
}

def build_wall_ghosts(state,previous_state,radii,margin=0.0005):
    positions = state[:,:3]
    previous_positions = previous_state[:,:3]
    omega = state.new_tensor([0.0,10.0*2.0*torch.pi/60.0,0.0])

    def wall_points(points):
        x,y,z = points.unbind(dim=1)
        radial = torch.sqrt(x*x+z*z).clamp_min(1e-8)
        cylinder = torch.stack([0.02*x/radial,y,0.02*z/radial],dim=1)
        lower = torch.stack([x,torch.full_like(y,-0.01),z],dim=1)
        upper = torch.stack([x,torch.full_like(y,0.01),z],dim=1)
        return [cylinder,lower,upper]

    def ghost_state(points,surface_points,wall_id):
        outward = torch.zeros_like(points)
        if wall_id == 0:
            outward[:,0] = surface_points[:,0]/0.02
            outward[:,2] = surface_points[:,2]/0.02
        elif wall_id == 1:
            outward[:,1] = -1.0
        else:
            outward[:,1] = 1.0
        signed_distance = ((points - surface_points)*outward).sum(dim=1,keepdim=True)
        reflected = (points +2.0*signed_distance.abs()*outward)
        wall_velocity = torch.cross(omega.expand_as(surface_points),surface_points,dim=1)
        return torch.cat([reflected,wall_velocity,omega.expand_as(points)], dim=1)

    current_points = wall_points(positions)
    previous_points = wall_points(previous_positions)
    radial = positions[:,[0,2]].norm(dim=1)
    gaps = [0.02 - radial - radii, positions[:, 1] + 0.01 - radii, 0.01 - positions[:, 1] - radii]
    owners = []
    wall_ids = []
    current_ghosts = []
    previous_ghosts = []
    selected_gaps = []
    for wall_id in range(3):
        selected = torch.where(gaps[wall_id] <= margin)[0]
        owners.append(selected)
        wall_ids.append(torch.full_like(selected,wall_id))
        current_ghosts.append(ghost_state(positions,current_points[wall_id],wall_id)[selected])
        previous_ghosts.append(ghost_state(previous_positions,previous_points[wall_id],wall_id)[selected])
        selected_gaps.append(gaps[wall_id][selected])
    return {
        "owners": torch.cat(owners),
        "wall_ids": torch.cat(wall_ids),
        "current": torch.cat(current_ghosts),
        "previous": torch.cat(previous_ghosts),
        "gaps": torch.cat(selected_gaps),
    }

    