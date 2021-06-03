FLAMEGPU_DEVICE_FUNCTION void vec3CrossProd(float &x, float &y, float &z, float x1, float y1, float z1, float x2, float y2, float z2) {
  x = (y1 * z2 - z1 * y2);
  y = (z1 * x2 - x1 * z2);
  z = (x1 * y2 - y1 * x2);
}
FLAMEGPU_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
  x /= divisor;
  y /= divisor;
  z /= divisor;
}
FLAMEGPU_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}
FLAMEGPU_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
  float length = vec3Length(x, y, z);
  vec3Div(x, y, z, length);
}

FLAMEGPU_AGENT_FUNCTION(ecm_ecm_interaction, MsgArray3D, MsgNone) {
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");

  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");

  // Agent grid position
  uint8_t agent_grid_i = FLAMEGPU->getVariable<uint8_t>("grid_i");
  uint8_t agent_grid_j = FLAMEGPU->getVariable<uint8_t>("grid_j");
  uint8_t agent_grid_k = FLAMEGPU->getVariable<uint8_t>("grid_k");
  
  // Agent velocity
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");

  
  // Elastinc constant of the ecm 
  const float k_elast = FLAMEGPU->getVariable<float>("k_elast");
  
  // Dumping constant of the ecm 
  const float d_dumping = FLAMEGPU->getVariable<float>("d_dumping");
  
  const float ECM_ECM_INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("ECM_ECM_INTERACTION_RADIUS");
  const float ECM_ECM_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("ECM_ECM_EQUILIBRIUM_DISTANCE");
  
  float agent_fx = 0.0;
  float agent_fy = 0.0;
  float agent_fz = 0.0; 
  float agent_f_extension = 0.0;
  float agent_f_compression = 0.0;
  float agent_elastic_energy = 0.0;
  
  float message_x = 0.0;
  float message_y = 0.0;
  float message_z = 0.0;
  int message_id = 0;
  float message_vx = 0.0;
  float message_vy = 0.0;
  float message_vz = 0.0;
  uint8_t message_grid_i = 0;
  uint8_t message_grid_j = 0;
  uint8_t message_grid_k = 0;

  // Initialize other variables
  float EPSILON = 0.0000000001;
  // cross product between agent-message velocity vectors and vector joining agents (direction)
  float cross_agent_vx_dir = 0.0;
  float cross_agent_vy_dir = 0.0;
  float cross_agent_vz_dir = 0.0;
  float cross_message_vx_dir = 0.0;
  float cross_message_vy_dir = 0.0;
  float cross_message_vz_dir = 0.0;
  // direction: the vector joining interacting agents
  float dir_x = 0.0; 
  float dir_y = 0.0; 
  float dir_z = 0.0; 
  float distance = 0.0; 
  // director cosines (with respect to global axis) of the direction vector
  float cos_x = 0.0;
  float cos_y = 0.0;
  float cos_z = 0.0;
  // dot product, determinant and angle (in radians) between agent velocity vector and direction vector
  float dot_agent_v_dir = 0.0;
  float det_agent_v_dir = 0.0;
  float angle_agent_v_dir = 0.0;
  float dot_message_v_dir = 0.0;
  float det_message_v_dir = 0.0;
  float angle_message_v_dir = 0.0;
  // relative speed between agents
  float relative_speed = 0.0;
  // total force between agents
  float total_f = 0.0;

  int conn = 0;
  int ct = 0;
  int DEBUG_PRINTING = FLAMEGPU->environment.getProperty<int>("DEBUG_PRINTING");


  //printf("Interaction agent %d [%d %d %d]\n", id, agent_grid_i, agent_grid_j, agent_grid_k);

  // Iterate location messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_grid_i, agent_grid_j, agent_grid_k)) {
    message_id = message.getVariable<int>("id");
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_z = message.getVariable<float>("z");
    message_grid_i = message.getVariable<uint8_t>("grid_i");
    message_grid_j = message.getVariable<uint8_t>("grid_j");
    message_grid_k = message.getVariable<uint8_t>("grid_k");

    conn = abs(agent_grid_i - message_grid_i) + abs(agent_grid_j - message_grid_j) + abs(agent_grid_k - message_grid_k);
    //printf("message id (for agent %d): %d [%d %d %d], conn = %d \n", message_id, id, message_grid_i, message_grid_j, message_grid_k, conn);
    
    if ((id != message_id) && (conn < 2)){

        ct++;
        dir_x = agent_x - message_x; 
        dir_y = agent_y - message_y; 
        dir_z = agent_z - message_z; 
        distance = vec3Length(dir_x, dir_y, dir_z);      
        
        message_vx = message.getVariable<float>("vx");
        message_vy = message.getVariable<float>("vy");
        message_vz = message.getVariable<float>("vz");

        cos_x = (1.0 * dir_x + 0.0 * dir_y + 0.0 * dir_z) / distance;
        cos_y = (0.0 * dir_x + 1.0 * dir_y + 0.0 * dir_z) / distance;
        cos_z = (0.0 * dir_x + 0.0 * dir_y + 1.0 * dir_z) / distance;

        dot_agent_v_dir = agent_vx * dir_x + agent_vy * dir_y + agent_vz * dir_z;
        vec3CrossProd(cross_agent_vx_dir, cross_agent_vy_dir, cross_agent_vz_dir, agent_vx, agent_vy, agent_vz, dir_x, dir_y, dir_z);
        det_agent_v_dir = vec3Length(cross_agent_vx_dir, cross_agent_vy_dir, cross_agent_vz_dir);
        if (fabsf(dot_agent_v_dir) > EPSILON) {
            angle_agent_v_dir = atan2f(det_agent_v_dir, dot_agent_v_dir);
        }
        else {
            angle_agent_v_dir = 0.0;
        }

        dot_message_v_dir = message_vx * dir_x + message_vy * dir_y + message_vz * dir_z;
        vec3CrossProd(cross_message_vx_dir, cross_message_vy_dir, cross_message_vz_dir, message_vx, message_vy, message_vz, dir_x, dir_y, dir_z);
        det_message_v_dir = vec3Length(cross_message_vx_dir, cross_message_vy_dir, cross_message_vz_dir);
        angle_message_v_dir = atan2f(det_message_v_dir, dot_message_v_dir);
        if (fabsf(dot_message_v_dir) > EPSILON) {
            angle_message_v_dir = atan2f(det_message_v_dir, dot_message_v_dir);
        }
        else {
            angle_message_v_dir = 0.0;
        }

        // relative speed <0 means particles are getting closer
        relative_speed = vec3Length(agent_vx, agent_vy, agent_vz) * cosf(angle_agent_v_dir) - vec3Length(message_vx, message_vy, message_vz) * cosf(angle_message_v_dir);
        // if total_f > 0, agents are attracted, if <0 agents are repelled
        total_f = (distance - ECM_ECM_EQUILIBRIUM_DISTANCE) * (k_elast)+d_dumping * relative_speed;

        if (total_f < 0) {
            agent_f_compression += total_f;
        }
        else {
            agent_f_extension += total_f;
        }

        agent_elastic_energy += 0.5 * (total_f * total_f) / k_elast;

        agent_fx += -1 * total_f * cos_x;
        agent_fy += -1 * total_f * cos_y;
        agent_fz += -1 * total_f * cos_z;

        if (DEBUG_PRINTING == 1) {
            printf("ECM interaction [id1: %d - id2: %d] agent_pos (%2.6f, %2.6f, %2.6f), message_pos (%2.6f, %2.6f, %2.6f)\n", id, message_id, agent_x, agent_y, agent_z, message_x, message_y, message_z);
            printf("ECM interaction id1: %d - id2: %d distance -> (%2.6f)\n", id, message_id, distance);
            printf("ECM interaction id1: %d - id2: %d total_f -> %2.6f (%2.6f , %2.6f, %2.6f)\n", id, message_id, total_f, agent_fx, agent_fy, agent_fy);
        }
    }
  }
  if (threadIdx.x%2 == 0) {
    //printf("Array3D for agent %d read %d messages!\n", id, ct);
  }
  //printf("Array3D for agent %d read %d messages! grid [%d %d %d], pos (%2.6f , %2.6f, %2.6f) \n", id, ct, agent_grid_i, agent_grid_j, agent_grid_k, agent_x, agent_y, agent_z);

  FLAMEGPU->setVariable<float>("fx", agent_fx);
  FLAMEGPU->setVariable<float>("fy", agent_fy);
  FLAMEGPU->setVariable<float>("fz", agent_fz);
  FLAMEGPU->setVariable<float>("f_extension", agent_f_extension);
  FLAMEGPU->setVariable<float>("f_compression", agent_f_compression);
  FLAMEGPU->setVariable<float>("elastic_energy", agent_elastic_energy);

  return ALIVE;
}