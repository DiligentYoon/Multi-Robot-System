import numpy as np

from task.utils.graph_utils import astar_search
from task.utils.transform_utils import world_to_local


class AgentRouter:
    def route_all(self,
                  map_info,
                  connectivity_graph,
                  robot_locations,
                  robot_angles,
                  robot_velocities,
                  robot_speeds,
                  obstacle_states,
                  num_obstacles,
                  neighbor_states,
                  num_neighbors,
                  assigned_rc,
                  cfg,
                  no_path_until_refresh: np.ndarray
                  ) -> dict:

        num_agent = len(robot_locations)

        follower_list         = []
        target_pos_list       = []
        end_pos_world_list    = []
        p_obs_list            = []
        p_agents_list         = []
        p_c_agent_list        = []
        v_c_agent_list        = []
        v_agents_local_list   = []
        connectivity_traj     = [[] for _ in range(num_agent)]
        routing_log_lines: list[str] = []

        for i in range(num_agent):
            pos_i = robot_locations[i]
            yaw_i = robot_angles[i]

            # Obstacle Info
            num_obs = num_obstacles[i]
            p_obs_list.append(obstacle_states[i, :num_obs])
            # Neighbor Agent Info
            num_nbrs = num_neighbors[i]
            p_agents_list.append(neighbor_states[i, :num_nbrs, :2])
            v_agents_local_list.append(neighbor_states[i, :num_nbrs, 2:])

            # ==== Target Agent Info ====
            # Root Node : Parent Node 존재 X
            # Leaf Node : CHild Node 존재 X
            # Reciprocal Connectivity Relationship:
            #   1. Parent Node : Child Node와 HOCBF 제약
            #   2. Child Node : Parent Node까지 A* Optimal Path
            #   3. Root Node : Child Node에 대해서 Only HOCBF제약
            #   4. Leaf Node : Parents Node에 대해서 Only A* Optimal Path

            #Proposed
            parent_id = connectivity_graph.get_parent(i)
            child_id  = connectivity_graph.get_child(i)

            if parent_id == -1:
                # Root Node : Only HOCBF
                pos_i_c = robot_locations[child_id]
                vel_i_c = robot_velocities[child_id]
                if pos_i_c.shape[0] > 1:
                    # MST는 여러개의 자식노드가 있을 수 있음 -> Closest로 지정
                    min_ids = np.argmin(np.linalg.norm(pos_i - pos_i_c, axis=1))
                    pos_i_cbf = pos_i_c[min_ids]
                    vel_i_cbf = vel_i_c[min_ids]
                else:
                    pos_i_cbf = pos_i_c.reshape(-1)
                    vel_i_cbf = vel_i_c.reshape(-1)
            elif child_id is None:
                # Leaf Node : Only A*
                pos_i_op = robot_locations[parent_id]
            else:
                # Other Node
                pos_i_op = robot_locations[parent_id]

                pos_i_c = robot_locations[child_id]
                vel_i_c = robot_velocities[child_id]
                if pos_i_c.shape[0] > 1:
                    # MST는 여러개의 자식노드가 있을 수 있음 -> Closest로 지정
                    min_ids = np.argmin(np.linalg.norm(pos_i - pos_i_c, axis=1))
                    pos_i_cbf = pos_i_c[min_ids]
                    vel_i_cbf = vel_i_c[min_ids]
                else:
                    pos_i_cbf = pos_i_c.reshape(-1)
                    vel_i_cbf = vel_i_c.reshape(-1)

            # Control Barrier Function Info for Backward Connectivity
            if child_id is None:
                # Leaf Node : CBF 적용 X, A*를 위한 Position만 할당
                p_p = world_to_local(w1=pos_i, w2=pos_i_op, yaw=yaw_i) # Parent for Connectivity A*
                p_c_agent_list.append(np.array([]))
                v_c_agent_list.append(np.array([]))
            else:
                # Child Node가 있는 Node들 {Root Node, Other Node}
                if parent_id == -1:
                    p_p = np.array([0, 0]) # Root Node don't have the parent node
                else:
                    p_p = world_to_local(w1=pos_i, w2=pos_i_op, yaw=yaw_i)  # Parent for Connectivity A*
                p_c = world_to_local(w1=pos_i, w2=pos_i_cbf, yaw=yaw_i)     # Child for Connectivity HOCBF
                v_c = world_to_local(w1=None, w2=vel_i_cbf, yaw=yaw_i)
                p_c_agent_list.append(p_c)
                v_c_agent_list.append(v_c)

            # Target Position Info & Forward Connectivity Info
            min_dist = np.linalg.norm(p_p)

            if (parent_id == -1):
                # 루트는 무조건 TARGET 쪽으로
                follower = False
                start_cell = map_info.world_to_grid_np(pos_i)
                end_cell = assigned_rc[i]
                end_world = map_info.grid_to_world_np(end_cell)
            else:
                min_dist = np.linalg.norm(p_p)
                overlap = (np.linalg.norm(assigned_rc[i] - assigned_rc[parent_id]) < 10)

                if (min_dist < cfg.d_max - 0.1) and (not overlap):
                    follower = False
                    start_cell = map_info.world_to_grid_np(pos_i)
                    end_cell = assigned_rc[i]
                    end_world = map_info.grid_to_world_np(end_cell)
                else:
                    follower = True
                    start_cell = map_info.world_to_grid_np(pos_i)
                    end_cell = map_info.world_to_grid_np(pos_i_op)
                    end_world = pos_i_op

            # 이후 A* 탐색 및 리스트 추가 로직은 동일하게 유지
            # ----- 라우팅 요약 로그 생성 -----
            if not follower:
                # 그냥 타깃으로 가는 경우: 이유는 안 붙임
                if parent_id == -1:
                    log_line = f"Agent {i}: --> TARGET (root)"
                else:
                    log_line = f"Agent {i}: --> TARGET"
            else:
                # follower = True 인 경우에만 "왜 follower가 되었는지" 이유를 상세히 남김
                reasons = []

                # 1) 부모 관계 (여기선 항상 parent_id != -1)
                reasons.append(f"parent={parent_id}")

                # 2) 거리 조건
                if min_dist >= cfg.d_max - 0.1:
                    reasons.append(f"dist_to_parent={min_dist:.2f} ≥ d_max-0.1")

                # 3) 타깃 중첩
                if overlap:
                    reasons.append("target cell overlaps with parent")

                if not reasons:
                    reasons.append("connectivity at risk")

                log_reason = "; ".join(reasons)
                log_line = f"Agent {i}: --> Agent {parent_id} | reason: {log_reason}"
            routing_log_lines.append(log_line)

            if no_path_until_refresh[i]:
                target_pos = np.array([0.0, 0.0])
                target_pos_list.append(target_pos)
                end_pos_world_list.append(end_world.reshape(-1))
                follower_list.append(follower)
                # 이 에이전트에 대해서는 더 계산할 것 없이 다음 에이전트로
                continue

            # A* Graph Search for Optimal Path
            look_ahead_distance = 0.1
            path_cells = astar_search(map_info,
                                      start_pos=np.flip(start_cell), # (row, col)
                                      end_pos=np.flip(end_cell),     # (row, col)
                                      agent_id=i)

            if path_cells is not None and len(path_cells) > 0:
                optimal_traj = map_info.grid_to_world_np(np.flip(np.array(path_cells), axis=1))
                connectivity_traj[i].append(optimal_traj)
                optimal_traj_local = world_to_local(w1=pos_i, w2=optimal_traj, yaw=yaw_i)
                distance_traj = np.linalg.norm(optimal_traj_local, axis=1)
                ids = np.argwhere(distance_traj >= look_ahead_distance)
                if ids.size > 0:
                    target_pos = optimal_traj_local[ids[0][0]]
                else:
                    target_pos = optimal_traj_local[-1]
            else:
                # 경로가 안 나왔을 때: world 좌표로 디버그 메시지만 출력
                no_path_until_refresh[i] = True
                print(
                    f"[A*] No valid path for agent {i}: "
                )
                # fallback: 제자리 명령
                target_pos = np.array([0.0, 0.0])

            target_pos_list.append(target_pos)
            end_pos_world_list.append(end_world.reshape(-1))
            follower_list.append(follower)

        if routing_log_lines:
            print("\n[Routing summary]")
            for line in routing_log_lines:
                print("  " + line)
            print()

        return {
            "target_pos_list"      : target_pos_list,
            "end_pos_world"        : np.array(end_pos_world_list),
            "follower_list"        : follower_list,
            "p_obs_list"           : p_obs_list,
            "p_agents_list"        : p_agents_list,
            "v_agents_local_list"  : v_agents_local_list,
            "p_c_agent_list"       : p_c_agent_list,
            "v_c_agent_list"       : v_c_agent_list,
            "connectivity_traj"    : connectivity_traj,
            "no_path_until_refresh": no_path_until_refresh,
        }
