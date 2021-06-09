FLAMEGPU_AGENT_FUNCTION(ecm_boundary_interaction, MsgNone, MsgNone) {
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");

  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  
  // Agent velocity
  float agent_vx = FLAMEGPU->getVariable<float>("vx");
  float agent_vy = FLAMEGPU->getVariable<float>("vy");
  float agent_vz = FLAMEGPU->getVariable<float>("vz");
  
  // Elastinc constant of the ecm 
  //const float k_elast = FLAMEGPU->getVariable<float>("k_elast");
  float k_elast = 0.0;
  
  // Dumping constant of the ecm 
  //const float d_dumping = FLAMEGPU->getVariable<float>("d_dumping");
  float d_dumping = 0.0;
  
  //Interaction with boundaries
  float boundary_fx = 0.0;
  float boundary_fy = 0.0;
  float boundary_fz = 0.0;
  float separation_x_pos = 0.0;
  float separation_x_neg = 0.0;
  float separation_y_pos = 0.0;
  float separation_y_neg = 0.0;
  float separation_z_pos = 0.0;
  float separation_z_neg = 0.0;
  const float ECM_BOUNDARY_INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("ECM_BOUNDARY_INTERACTION_RADIUS");
  const float ECM_BOUNDARY_EQUILIBRIUM_DISTANCE = FLAMEGPU->environment.getProperty<float>("ECM_BOUNDARY_EQUILIBRIUM_DISTANCE");
  float EPSILON = 0.0000000001;

  // Get position of the boundaries
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);
  
  // Get displacement rate of the boundaries
  const float DISP_RATE_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",0);
  const float DISP_RATE_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",1);
  const float DISP_RATE_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",2);
  const float DISP_RATE_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",3);
  const float DISP_RATE_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",4);
  const float DISP_RATE_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("DISP_RATES_BOUNDARIES",5);

  // Check for ecm-boundary separations
  separation_x_pos = fabsf(agent_x - COORD_BOUNDARY_X_POS);
  separation_x_neg = fabsf(agent_x - COORD_BOUNDARY_X_NEG);
  separation_y_pos = fabsf(agent_y - COORD_BOUNDARY_Y_POS);
  separation_y_neg = fabsf(agent_y - COORD_BOUNDARY_Y_NEG);
  separation_z_pos = fabsf(agent_z - COORD_BOUNDARY_Z_POS);
  separation_z_neg = fabsf(agent_z - COORD_BOUNDARY_Z_NEG);

  
  

  if (separation_x_pos < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_x_pos) > EPSILON) {
      boundary_fx +=  (separation_x_pos - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE) * (k_elast) - d_dumping * (agent_vx - DISP_RATE_BOUNDARY_X_POS);      
  }
  if (separation_x_neg < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_x_neg) > EPSILON) {
      boundary_fx +=  -1* (separation_x_neg - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE) * (k_elast) - d_dumping * (agent_vx - DISP_RATE_BOUNDARY_X_NEG);      
  }
  if (separation_y_pos < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_y_pos) > EPSILON) {
      boundary_fy +=  (separation_y_pos - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE) * (k_elast) - d_dumping * (agent_vy - DISP_RATE_BOUNDARY_Y_POS);      
  }
  if (separation_y_neg < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_y_neg) > EPSILON) {
      boundary_fy +=  -1* (separation_y_neg - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE) * (k_elast) - d_dumping * (agent_vy - DISP_RATE_BOUNDARY_Y_NEG);      
  }
  if (separation_z_pos < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_z_pos) > EPSILON) {
      boundary_fz +=  (separation_z_pos - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE) * (k_elast) - d_dumping * (agent_vz - DISP_RATE_BOUNDARY_Z_POS);      
  }
  if (separation_z_neg < (ECM_BOUNDARY_INTERACTION_RADIUS) && fabsf(separation_z_neg) > EPSILON) {
      boundary_fz +=  -1* (separation_z_neg - ECM_BOUNDARY_EQUILIBRIUM_DISTANCE) * (k_elast) - d_dumping * (agent_vz - DISP_RATE_BOUNDARY_Z_NEG);      
  }

  FLAMEGPU->setVariable<float>("boundary_fx", boundary_fx);
  FLAMEGPU->setVariable<float>("boundary_fy", boundary_fy);
  FLAMEGPU->setVariable<float>("boundary_fz", boundary_fz);

  return ALIVE;
}