FLAMEGPU_HOST_DEVICE_FUNCTION void boundPosition(float &x, float &y, float &z, 
        uint8_t &cxpos, uint8_t &cxneg, uint8_t &cypos, uint8_t &cyneg, uint8_t &czpos, uint8_t &czneg, 
        const float bxpos, const float bxneg, const float bypos, const float byneg, const float bzpos, const float bzneg,
        const int &clamp_on) {
        
  if (x > bxpos){
      x = bxpos;
      if (clamp_on == 1) {
         cxpos = 1;
      }
  }    
  if (x < bxneg){
      x = bxneg;
      if (clamp_on == 1) {
         cxneg = 1;
      }
  }
  if (y > bypos){
      y = bypos;
      if (clamp_on == 1) {
         cypos = 1;
      }
  }    
  if (y < byneg){
      y = byneg;
      if (clamp_on == 1) {
         cyneg = 1;
      }
  }
  if (z > bzpos){
      z = bzpos;
      if (clamp_on == 1) {
         czpos = 1;
      }
  }    
  if (z < bzneg){
      z = bzneg;
      if (clamp_on == 1) {
         czneg = 1;
      }
  }
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

  
  if (DEBUG_PRINTING == 1){
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
  
  agent_vx += (agent_fx / mass) * DELTA_TIME;
  agent_vy += (agent_fy / mass) * DELTA_TIME;
  agent_vz += (agent_fz / mass) * DELTA_TIME;
  
  agent_x += agent_vx * DELTA_TIME;
  agent_y += agent_vy * DELTA_TIME;
  agent_z += agent_vz * DELTA_TIME;


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
  
  if (clamped_bx_pos == 1){
    agent_x = COORD_BOUNDARY_X_POS;
    agent_vx = DISP_RATE_BOUNDARY_X_POS;
  }
  if (clamped_bx_neg == 1){
    agent_x = COORD_BOUNDARY_X_NEG;
    agent_vx = DISP_RATE_BOUNDARY_X_NEG;
  }
  if (clamped_by_pos == 1){
    agent_y = COORD_BOUNDARY_Y_POS;
    agent_vy = DISP_RATE_BOUNDARY_Y_POS;
  }
  if (clamped_by_neg == 1){
    agent_y = COORD_BOUNDARY_Y_NEG;
    agent_vy = DISP_RATE_BOUNDARY_Y_NEG;
  }
  if (clamped_bz_pos == 1){
    agent_z = COORD_BOUNDARY_Z_POS;
    agent_vz = DISP_RATE_BOUNDARY_Z_POS;
  }
  if (clamped_bx_neg == 1){
    agent_z = COORD_BOUNDARY_Z_NEG;
    agent_vz = DISP_RATE_BOUNDARY_Z_NEG;
  }
    
  boundPosition(agent_x, agent_y, agent_z, 
                clamped_bx_pos, clamped_bx_neg, clamped_by_pos, clamped_by_neg, clamped_bz_pos, clamped_bz_neg, 
                COORD_BOUNDARY_X_POS, COORD_BOUNDARY_X_NEG, COORD_BOUNDARY_Y_POS, COORD_BOUNDARY_Y_NEG, COORD_BOUNDARY_Z_POS, COORD_BOUNDARY_Z_NEG,
                CLAMP_AGENT_TOUCHING_BOUNDARY);

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