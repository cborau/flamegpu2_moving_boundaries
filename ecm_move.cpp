FLAMEGPU_HOST_DEVICE_FUNCTION void boundPosition(int id, float &x, float &y, float &z, 
        uint8_t &cxpos, uint8_t &cxneg, uint8_t &cypos, uint8_t &cyneg, uint8_t &czpos, uint8_t &czneg, 
        const float bxpos, const float bxneg, const float bypos, const float byneg, const float bzpos, const float bzneg,
        const int clamp_on, const float ecm_boundary_equilibrium_distance) {
    
  //if (id == 9 || id = 10) {
  //    printf("Boundposition ANTES agent %d position %2.4f, %2.4f, %2.4f ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n", id, x, y, z, bxpos, bxneg, bypos, byneg, bzpos, bzneg, cxpos, cxneg, cypos, cyneg, czpos, czneg);
  //}
  float EPSILON = 0.00000001;

  if (cxpos == 1) {
      x = bxpos - ecm_boundary_equilibrium_distance; // redundant. Could say "do nothing"
  } else {
      if (x > bxpos || fabsf(x - bxpos) < ecm_boundary_equilibrium_distance + EPSILON) {
          x = bxpos - ecm_boundary_equilibrium_distance;
          if (clamp_on == 1) {
              cxpos = 1;
          }
      }
  }
     
  if (cxneg == 1) {
      x = bxneg + ecm_boundary_equilibrium_distance;
  } else {
      if (x < bxneg || fabsf(x - bxneg) < ecm_boundary_equilibrium_distance + EPSILON) {
          x = bxneg + ecm_boundary_equilibrium_distance;
          if (clamp_on == 1) {
              cxneg = 1;
          }
      }
  }

  if (cypos == 1) {
      y = bypos - ecm_boundary_equilibrium_distance;
  } else {
      if (y > bypos || fabsf(y - bypos) < ecm_boundary_equilibrium_distance + EPSILON) {
          y = bypos - ecm_boundary_equilibrium_distance;
          if (clamp_on == 1) {
              cypos = 1;
          }
      }
  }
  
  if (cyneg == 1) {
      y = byneg + ecm_boundary_equilibrium_distance;
  } else {
      if (y < byneg || fabsf(y - byneg) < ecm_boundary_equilibrium_distance + EPSILON) {
          y = byneg + ecm_boundary_equilibrium_distance;
          if (clamp_on == 1) {
              cyneg = 1;
          }
      }
  }


  if (czpos == 1) {
      z = bzpos - ecm_boundary_equilibrium_distance;
  } else {
      if (z > bzpos || fabsf(z - bzpos) < ecm_boundary_equilibrium_distance + EPSILON) {
          z = bzpos - ecm_boundary_equilibrium_distance;
          if (clamp_on == 1) {
              czpos = 1;
          }
      }
  }
   
  if (czneg == 1) {
      z = bzneg + ecm_boundary_equilibrium_distance;
  }
  else {
      if (z < bzneg || fabsf(z - bzneg) < ecm_boundary_equilibrium_distance + EPSILON) {
          z = bzneg + ecm_boundary_equilibrium_distance;
          if (clamp_on == 1) {
              czneg = 1;
          }
      }
  }

  //if (id == 9 || id = 10) {
  //   printf("Boundposition DESPUES agent %d position %2.4f, %2.4f, %2.4f ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n", id, x, y, z, bxpos, bxneg, bypos, byneg, bzpos, bzneg, cxpos, cxneg, cypos, cyneg, czpos, czneg);
  //}
}
FLAMEGPU_AGENT_FUNCTION(ecm_move, MsgNone, MsgNone) {
  
  int id = FLAMEGPU->getVariable<int>("id");
  //Agent position vector
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");

  int DEBUG_PRINTING = FLAMEGPU->environment.getProperty<int>("DEBUG_PRINTING");

  // Agent velocity
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");
  
  // Agent clamps
  uint8_t clamped_bx_pos = FLAMEGPU->getVariable<uint8_t>("clamped_bx_pos");
  uint8_t clamped_bx_neg = FLAMEGPU->getVariable<uint8_t>("clamped_bx_neg");
  uint8_t clamped_by_pos = FLAMEGPU->getVariable<uint8_t>("clamped_by_pos");
  uint8_t clamped_by_neg = FLAMEGPU->getVariable<uint8_t>("clamped_by_neg");
  uint8_t clamped_bz_pos = FLAMEGPU->getVariable<uint8_t>("clamped_bz_pos");
  uint8_t clamped_bz_neg = FLAMEGPU->getVariable<uint8_t>("clamped_bz_neg");
   
  // Mass of the ecm agent
  const float mass = FLAMEGPU->getVariable<float>("mass");

  //Forces acting on the agent
  float agent_fx = FLAMEGPU->getVariable<float>("fx");
  float agent_fy = FLAMEGPU->getVariable<float>("fy");
  float agent_fz = FLAMEGPU->getVariable<float>("fz");
  float agent_boundary_fx = FLAMEGPU->getVariable<float>("boundary_fx");
  float agent_boundary_fy = FLAMEGPU->getVariable<float>("boundary_fy");
  float agent_boundary_fz = FLAMEGPU->getVariable<float>("boundary_fz");
  
 
  //Add the force coming from the boundaries
  agent_fx += agent_boundary_fx;
  agent_fy += agent_boundary_fy;
  agent_fz += agent_boundary_fz;

  
  if (DEBUG_PRINTING == 1 && (id == 9 || id == 10 || id == 14 || id == 25 || id == 26 || id == 29 || id == 30)){
    printf("ECM move ID: %d clamps before -> (%d, %d, %d, %d, %d, %d)\n", id, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);
    printf("ECM move ID: %d pos -> (%2.6f, %2.6f, %2.6f)\n", id, agent_x, agent_y, agent_z);
    printf("ECM move ID: %d vel -> (%2.6f, %2.6f, %2.6f)\n", id, agent_vx, agent_vy, agent_vz);
    printf("ECM move ID: %d f -> (%2.6f, %2.6f, %2.6f)\n", id, agent_fx, agent_fy, agent_fz);
    printf("ECM move ID: %d bf -> (%2.6f, %2.6f, %2.6f)\n", id, agent_boundary_fx, agent_boundary_fy, agent_boundary_fz);
    printf("ECM move ID: %d f after -> (%2.6f, %2.6f, %2.6f)\n", id, agent_fx, agent_fy, agent_fz);
  }

  //Get the new position and velocity: 
  // a(t) = f(t) / m;
  // v(t) = v(t-1) + a(t) * dt; 
  // x(t) = x(t-1) + v(t) * dt
  const float DELTA_TIME = FLAMEGPU->environment.getProperty<float>("DELTA_TIME");
  //Bound the position within the environment   
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);
  const float DISP_RATE_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",0);
  const float DISP_RATE_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",1);
  const float DISP_RATE_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",2);
  const float DISP_RATE_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",3);
  const float DISP_RATE_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",4);
  const float DISP_RATE_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",5);
  const int CLAMP_AGENT_TOUCHING_BOUNDARY = FLAMEGPU->environment.getProperty<int>("CLAMP_AGENT_TOUCHING_BOUNDARY");
  const int ALLOW_AGENT_SLIDING_X_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 0);
  const int ALLOW_AGENT_SLIDING_X_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 1);
  const int ALLOW_AGENT_SLIDING_Y_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 2);
  const int ALLOW_AGENT_SLIDING_Y_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 3);
  const int ALLOW_AGENT_SLIDING_Z_POS = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 4);
  const int ALLOW_AGENT_SLIDING_Z_NEG = FLAMEGPU->environment.getProperty<int>("ALLOW_AGENT_SLIDING", 5);
  const float ECM_BOUNDARY_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("ECM_BOUNDARY_EQUILIBRIUM_DISTANCE");

  float prev_agent_x = agent_x;
  float prev_agent_y = agent_y;
  float prev_agent_z = agent_z;
   
  if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) {
      agent_vx += (agent_fx / mass) * DELTA_TIME;
      agent_x += agent_vx * DELTA_TIME;
  }

  if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) {
      agent_vy += (agent_fy / mass) * DELTA_TIME;
      agent_y += agent_vy * DELTA_TIME;
  }
  
  if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
      agent_vz += (agent_fz / mass) * DELTA_TIME;
      agent_z += agent_vz * DELTA_TIME;
  }
  
  //if (id == 9 || id == 10 || id == 13 || id == 14 || id == 25 || id == 26 || id == 29 || id == 30) {
  if (id == 11 || id == 12 || id == 18) {
     printf("agent %d position ANTES (%2.4f, %2.4f, %2.4f) ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n",id, agent_x,agent_y,agent_z, COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);
  }
  
  
  if (clamped_bx_pos == 1){
    agent_x = COORD_BOUNDARY_X_POS - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vx = DISP_RATE_BOUNDARY_X_POS;
    if (ALLOW_AGENT_SLIDING_X_POS == 0) {
        if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) { // this must be checked to avoid overwriting when agent is clamped to multiple boundaries
            agent_vy = 0.0;
            agent_y = prev_agent_y;
        }
        if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
            agent_vz = 0.0;
            agent_z = prev_agent_z;
        }
    }
  }
  if (clamped_bx_neg == 1){
    agent_x = COORD_BOUNDARY_X_NEG + ECM_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vx = DISP_RATE_BOUNDARY_X_NEG;
    if (ALLOW_AGENT_SLIDING_X_NEG == 0) {
        if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) { 
            agent_vy = 0.0;
            agent_y = prev_agent_y;
        }
        if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
            agent_vz = 0.0;
            agent_z = prev_agent_z;
        }
    }
  }
  if (clamped_by_pos == 1){
    agent_y = COORD_BOUNDARY_Y_POS - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vy = DISP_RATE_BOUNDARY_Y_POS;
    if (ALLOW_AGENT_SLIDING_Y_POS == 0) {
        if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) { 
            agent_vx = 0.0;
            agent_x = prev_agent_x;
        }
        if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
            agent_vz = 0.0;
            agent_z = prev_agent_z;
        }
    }
  }
  if (clamped_by_neg == 1){
    agent_y = COORD_BOUNDARY_Y_NEG + ECM_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vy = DISP_RATE_BOUNDARY_Y_NEG;
    if (ALLOW_AGENT_SLIDING_Y_NEG == 0) {
        if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) {
            agent_vx = 0.0;
            agent_x = prev_agent_x;
        }
        if ((clamped_bz_pos == 0) && (clamped_bz_neg == 0)) {
            agent_vz = 0.0;
            agent_z = prev_agent_z;
        }
    }
  }
  if (clamped_bz_pos == 1){
    agent_z = COORD_BOUNDARY_Z_POS - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vz = DISP_RATE_BOUNDARY_Z_POS;
    if (ALLOW_AGENT_SLIDING_Z_POS == 0) {
        if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) {
            agent_vx = 0.0;
            agent_x = prev_agent_x;
        }
        if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) {
            agent_vy = 0.0;
            agent_y = prev_agent_y;
        }
    }
  }
  if (clamped_bz_neg == 1){
    agent_z = COORD_BOUNDARY_Z_NEG + ECM_BOUNDARY_EQUILIBRIUM_DISTANCE;
    agent_vz = DISP_RATE_BOUNDARY_Z_NEG;
    if (ALLOW_AGENT_SLIDING_Z_NEG == 0) {
        if ((clamped_bx_pos == 0) && (clamped_bx_neg == 0)) {
            agent_vx = 0.0;
            agent_x = prev_agent_x;
        }
        if ((clamped_by_pos == 0) && (clamped_by_neg == 0)) {
            agent_vy = 0.0;
            agent_y = prev_agent_y;
        }
    }
  }
  
  //if (id == 9 || id == 10 || id == 13 || id == 14 || id == 25 || id == 26 || id == 29 || id == 30) {
  if (id == 11 || id == 12 || id == 18) {
      printf("agent %d position EN MEDIO (%2.4f, %2.4f, %2.4f) ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n", id, agent_x, agent_y, agent_z, COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);
  }
   
  boundPosition(id,agent_x, agent_y, agent_z, 
                clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg, 
                COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG,
                CLAMP_AGENT_TOUCHING_BOUNDARY, ECM_BOUNDARY_EQUILIBRIUM_DISTANCE);

  
  //if (id == 9 || id == 10 || id == 13 || id == 14 || id == 25 || id == 26 || id == 29 || id == 30) {
  if (id == 11 || id == 12 || id == 18) {
      printf("agent %d position DESPUES (%2.4f, %2.4f, %2.4f) ->  boundary pos: [%2.4f, %2.4f, %2.4f, %2.4f, %2.4f, %2.4f], clamping: [%d, %d, %d, %d, %d, %d] \n", id, agent_x, agent_y, agent_z, COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);
  }
  
  //printf("ECM move ID: %d clamps after -> (%d, %d, %d, %d, %d, %d)\n", id, clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg);

  //Update the agents position and velocity
  FLAMEGPU->setVariable<float>("x",agent_x);
  FLAMEGPU->setVariable<float>("y",agent_y);
  FLAMEGPU->setVariable<float>("z",agent_z);
  FLAMEGPU->setVariable<float>("vx",agent_vx);
  FLAMEGPU->setVariable<float>("vy",agent_vy);
  FLAMEGPU->setVariable<float>("vz",agent_vz);
  FLAMEGPU->setVariable<uint8_t>("clamped_bx_pos", clamped_bx_pos);
  FLAMEGPU->setVariable<uint8_t>("clamped_bx_neg", clamped_bx_neg);
  FLAMEGPU->setVariable<uint8_t>("clamped_by_pos", clamped_by_pos);
  FLAMEGPU->setVariable<uint8_t>("clamped_by_neg", clamped_by_neg);
  FLAMEGPU->setVariable<uint8_t>("clamped_bz_pos", clamped_bz_pos);
  FLAMEGPU->setVariable<uint8_t>("clamped_bz_neg", clamped_bz_neg);

  return ALIVE;
}