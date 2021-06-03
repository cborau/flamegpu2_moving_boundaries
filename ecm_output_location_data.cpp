FLAMEGPU_AGENT_FUNCTION(ecm_output_location_data, MsgNone, MsgSpatial3D) {
  // Output each ECM agent's location (and other visible properties if implemented)
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
  FLAMEGPU->message_out.setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx"));
  FLAMEGPU->message_out.setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy"));
  FLAMEGPU->message_out.setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz"));
  return ALIVE;
  }