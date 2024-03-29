from pyflamegpu import *
import sys, random, math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sns.set()

# Set whether to run single model or ensemble, agent population size, and simulation steps 
ENSEMBLE = False;
ENSEMBLE_RUNS = 0;
ECM_POPULATION_SIZE = 1000; 
STEPS = 400;
# Change to false if pyflamegpu has not been built with visualisation support
VISUALISATION = True;
DEBUG_PRINTING = False;

# Interaction and mechanical parameters
TIME_STEP = 0.05; # seconds
BOUNDARY_COORDS = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]; #+X,-X,+Y,-Y,+Z,-Z
BOUNDARY_DISP_RATES = [0.0, 0.0, 0.075, 0.0, 0.0, 0.0]; # units/second
ECM_ECM_INTERACTION_RADIUS = 0.25;
#ECM_ECM_EQUILIBRIUM_DISTANCE = 0.45;
ECM_ECM_EQUILIBRIUM_DISTANCE = (BOUNDARY_COORDS[0] - BOUNDARY_COORDS[1])  / int(round(ECM_POPULATION_SIZE ** (1.0 / 3.0)))
print("ECM_ECM_EQUILIBRIUM_DISTANCE: ", ECM_ECM_EQUILIBRIUM_DISTANCE)
ECM_BOUNDARY_INTERACTION_RADIUS = 0.2;
ECM_BOUNDARY_EQUILIBRIUM_DISTANCE = 0.1;

ECM_K_ELAST = 1.0;
ECM_D_DUMPING = 5.2;
ECM_MASS = 1.0;

CLAMP_AGENT_TOUCHING_BOUNDARY = 1;

#MAX_SEARCH_RADIUS = max([ECM_ECM_INTERACTION_RADIUS, ECM_BOUNDARY_INTERACTION_RADIUS]);
MAX_SEARCH_RADIUS = 2.0; # this strongly affects the number of bins and therefore the memory allocated for simulations (more bins -> more memory -> faster (in theory))


# Other simulation parameters:
MAX_EXPECTED_BOUNDARY_POS = max(BOUNDARY_DISP_RATES) * STEPS * TIME_STEP + 1.0;
MIN_EXPECTED_BOUNDARY_POS = min(BOUNDARY_DISP_RATES) * STEPS * TIME_STEP - 1.0;

print("Max expected boundary position: ", MAX_EXPECTED_BOUNDARY_POS);
print("Min expected boundary position: ", MIN_EXPECTED_BOUNDARY_POS);


"""
  FLAME GPU 2 implementation of mechanical assays via moving boundaries and extracellular matrix (ecm) agents by using spatial3D messaging.
"""


bcorner_output_location_data_file = "bcorner_output_location_data.cpp";
bcorner_move_file = "bcorner_move.cpp";

"""
  ECM
  ecm_output_location_data agent function for ECM agents, which outputs publicly visible properties to a message list
"""
ecm_output_location_data_file = "ecm_output_location_data.cpp";
ecm_boundary_interaction_file = "ecm_boundary_interaction.cpp";
ecm_ecm_interaction_file = "ecm_ecm_interaction.cpp";

"""
  ecm_move agent function for ECM agents
"""
ecm_move_file = "ecm_move.cpp";

model = pyflamegpu.ModelDescription("ECM_Moving_Boundaries");


"""
  GLOBALS
"""
env = model.Environment();
# Population size to generate, if no agents are loaded from disk
env.newPropertyUInt("ECM_POPULATION_TO_GENERATE", ECM_POPULATION_SIZE);
env.newPropertyUInt("CURRENT_ID", 0);

# Number of steps to simulate
env.newPropertyUInt("STEPS", STEPS);
# Time increment (seconds)
env.newPropertyFloat("DELTA_TIME", TIME_STEP);

# ------------------------------------------------------
# ECM BEHAVIOUR 
# ------------------------------------------------------
env.newPropertyFloat("ECM_ECM_INTERACTION_RADIUS", ECM_ECM_INTERACTION_RADIUS);
# Equilibrium radius at which elastic force is 0. 
# If ECM_ECM_INTERACTION_RADIUS > ECM_ECM_EQUILIBRIUM_DISTANCE: both repulsion/atraction can occur
# If ECM_ECM_INTERACTION_RADIUS <= ECM_ECM_EQUILIBRIUM_DISTANCE: only repulsion can occur
env.newPropertyFloat("ECM_ECM_EQUILIBRIUM_DISTANCE", ECM_ECM_EQUILIBRIUM_DISTANCE);
# Mechanical parameters
env.newPropertyFloat("ECM_K_ELAST", ECM_K_ELAST);
env.newPropertyFloat("ECM_D_DUMPING", ECM_D_DUMPING);
env.newPropertyFloat("ECM_MASS", ECM_MASS);

# ------------------------------------------------------
# BOUNDARY BEHAVIOUR 
# ------------------------------------------------------
# Boundaries position
bcs = [BOUNDARY_COORDS[0], BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[3], BOUNDARY_COORDS[4], BOUNDARY_COORDS[5]];  #+X,-X,+Y,-Y,+Z,-Z
env.newPropertyArrayFloat("COORDS_BOUNDARIES", 6, bcs);

# Boundaries displacement rate (units/second). 
bdrs = [BOUNDARY_DISP_RATES[0], BOUNDARY_DISP_RATES[1], BOUNDARY_DISP_RATES[2], BOUNDARY_DISP_RATES[3], BOUNDARY_DISP_RATES[4], BOUNDARY_DISP_RATES[5]]; #+X,-X,+Y,-Y,+Z,-Z
# e.g. DISP_BOUNDARY_X_POS = 0.1 means that this boundary moves 0.1 units per second towards +X
env.newPropertyArrayFloat("DISP_RATES_BOUNDARIES", 6,  bdrs); 

# Boundary-Agent behaviour
env.newPropertyUInt("CLAMP_AGENT_TOUCHING_BOUNDARY", CLAMP_AGENT_TOUCHING_BOUNDARY);
env.newPropertyFloat("ECM_BOUNDARY_INTERACTION_RADIUS", ECM_BOUNDARY_INTERACTION_RADIUS);
env.newPropertyFloat("ECM_BOUNDARY_EQUILIBRIUM_DISTANCE", ECM_BOUNDARY_EQUILIBRIUM_DISTANCE);


# Other globals
env.newPropertyFloat("PI", 3.1415);
env.newPropertyUInt("DEBUG_PRINTING", DEBUG_PRINTING);


"""
  Location messages
"""
bcorner_location_message = model.newMessageSpatial3D("bcorner_location_message");
# Set the range and bounds.
bcorner_location_message.setRadius(1.0); #corners are not actually interacting with anything
bcorner_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS);
bcorner_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS);
# A message to hold the location of an agent.
bcorner_location_message.newVariableInt("id");


ecm_location_message = model.newMessageSpatial3D("ecm_location_message");
# Set the range and bounds.
ecm_location_message.setRadius(MAX_SEARCH_RADIUS); 
ecm_location_message.setMin(MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS, MIN_EXPECTED_BOUNDARY_POS);
ecm_location_message.setMax(MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS, MAX_EXPECTED_BOUNDARY_POS);
# A message to hold the location of an agent.
ecm_location_message.newVariableInt("id");
ecm_location_message.newVariableFloat("vx");
ecm_location_message.newVariableFloat("vy");
ecm_location_message.newVariableFloat("vz");


"""
  Boundary corner agent
"""
bcorner_agent = model.newAgent("BCORNER");
bcorner_agent.newVariableInt("id");
bcorner_agent.newVariableFloat("x");
bcorner_agent.newVariableFloat("y");
bcorner_agent.newVariableFloat("z");

bcorner_agent.newRTCFunctionFile("bcorner_output_location_data", bcorner_output_location_data_file).setMessageOutput("bcorner_location_message");
#bcorner_agent.newRTCFunction("bcorner_move", bcorner_move);
bcorner_agent.newRTCFunctionFile("bcorner_move", bcorner_move_file);
    
"""
  ECM agent
"""
ecm_agent = model.newAgent("ECM");
ecm_agent.newVariableInt("id");
ecm_agent.newVariableFloat("x");
ecm_agent.newVariableFloat("y");
ecm_agent.newVariableFloat("z");
ecm_agent.newVariableFloat("vx");
ecm_agent.newVariableFloat("vy");
ecm_agent.newVariableFloat("vz");
ecm_agent.newVariableFloat("fx");
ecm_agent.newVariableFloat("fy");
ecm_agent.newVariableFloat("fz");
ecm_agent.newVariableFloat("k_elast");
ecm_agent.newVariableFloat("d_dumping");
ecm_agent.newVariableFloat("mass");
ecm_agent.newVariableFloat("boundary_fx");
ecm_agent.newVariableFloat("boundary_fy");
ecm_agent.newVariableFloat("boundary_fz");
ecm_agent.newVariableFloat("f_extension");
ecm_agent.newVariableFloat("f_compression");
ecm_agent.newVariableFloat("elastic_energy");
ecm_agent.newVariableUInt8("clamped_bx_pos");
ecm_agent.newVariableUInt8("clamped_bx_neg");
ecm_agent.newVariableUInt8("clamped_by_pos");
ecm_agent.newVariableUInt8("clamped_by_neg");
ecm_agent.newVariableUInt8("clamped_bz_pos");
ecm_agent.newVariableUInt8("clamped_bz_neg");

ecm_agent.newRTCFunctionFile("ecm_output_location_data", ecm_output_location_data_file).setMessageOutput("ecm_location_message");
ecm_agent.newRTCFunctionFile("ecm_boundary_interaction", ecm_boundary_interaction_file);
ecm_agent.newRTCFunctionFile("ecm_ecm_interaction", ecm_ecm_interaction_file).setMessageInput("ecm_location_message");
ecm_agent.newRTCFunctionFile("ecm_move", ecm_move_file);



"""
  Population initialisation functions
"""
# This class is used to ensure that corner agents are assigned the first 8 ids
class initAgentPopulations(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
    # BOUNDARY CORNERS
    current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID");
    coord_boundary = FLAMEGPU.environment.getPropertyArrayFloat("COORDS_BOUNDARIES")
    coord_boundary_x_pos = coord_boundary[0];
    coord_boundary_x_neg = coord_boundary[1];
    coord_boundary_y_pos = coord_boundary[2];
    coord_boundary_y_neg = coord_boundary[3];
    coord_boundary_z_pos = coord_boundary[4];
    coord_boundary_z_neg = coord_boundary[5];
    print("CORNERS:");
    print("current_id:", current_id);
    
    for i in range(1,9):
      instance = FLAMEGPU.agent("BCORNER").newAgent();
      instance.setVariableInt("id", current_id+i);
      if i == 1 :
      # +x,+y,+z
        instance.setVariableFloat("x", coord_boundary_x_pos);
        instance.setVariableFloat("y", coord_boundary_y_pos);
        instance.setVariableFloat("z", coord_boundary_z_pos);           
      elif i == 2 :
      # -x,+y,+z
        instance.setVariableFloat("x", coord_boundary_x_neg);
        instance.setVariableFloat("y", coord_boundary_y_pos);
        instance.setVariableFloat("z", coord_boundary_z_pos); 
      elif i == 3 :
      # -x,-y,+z
        instance.setVariableFloat("x", coord_boundary_x_neg);
        instance.setVariableFloat("y", coord_boundary_y_neg);
        instance.setVariableFloat("z", coord_boundary_z_pos);
      elif i == 4 :
      # +x,-y,+z
        instance.setVariableFloat("x", coord_boundary_x_pos);
        instance.setVariableFloat("y", coord_boundary_y_neg);
        instance.setVariableFloat("z", coord_boundary_z_pos);
      elif i == 5 :
      # +x,+y,-z
        instance.setVariableFloat("x", coord_boundary_x_pos);
        instance.setVariableFloat("y", coord_boundary_y_pos);
        instance.setVariableFloat("z", coord_boundary_z_neg);
      elif i == 6 :
      # -x,+y,-z
        instance.setVariableFloat("x", coord_boundary_x_neg);
        instance.setVariableFloat("y", coord_boundary_y_pos);
        instance.setVariableFloat("z", coord_boundary_z_neg);
      elif i == 7 :
      # -x,-y,-z
        instance.setVariableFloat("x", coord_boundary_x_neg);
        instance.setVariableFloat("y", coord_boundary_y_neg);
        instance.setVariableFloat("z", coord_boundary_z_neg); 
      elif i == 8 :
      # +x,-y,-z
        instance.setVariableFloat("x", coord_boundary_x_pos);
        instance.setVariableFloat("y", coord_boundary_y_neg);
        instance.setVariableFloat("z", coord_boundary_z_neg);   
      else :
        sys.exit("Bad initialization of boundary corners!");

    FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", 8);
  
    # ECM
    populationSize = FLAMEGPU.environment.getPropertyUInt("ECM_POPULATION_TO_GENERATE");
    min_pos = -1.0;
    max_pos = 1.0;
    min_speed = 0.0;
    max_speed = 0.0;
    k_elast = FLAMEGPU.environment.getPropertyFloat("ECM_K_ELAST");
    d_dumping = FLAMEGPU.environment.getPropertyFloat("ECM_D_DUMPING");
    mass = FLAMEGPU.environment.getPropertyFloat("ECM_MASS");
    current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID");
    current_id += 1;
    print("ECM:");
    print("current_id:", current_id);
    agents_per_dir = [int(round(populationSize ** (1.0 / 3.0))), int(round(populationSize ** (1.0 / 3.0))), int(round(populationSize ** (1.0 / 3.0)))];
    print("agents per dir", agents_per_dir);
    offset = 0.1; #ECM_BOUNDARY_INTERACTION_RADIUS
    coords_x = np.linspace(BOUNDARY_COORDS[1] + offset, BOUNDARY_COORDS[0] - offset, agents_per_dir[0]);
    coords_y = np.linspace(BOUNDARY_COORDS[3] + offset, BOUNDARY_COORDS[2] - offset, agents_per_dir[1]);
    coords_z = np.linspace(BOUNDARY_COORDS[5] + offset, BOUNDARY_COORDS[4] - offset, agents_per_dir[2]);
    print(coords_x);
    i = -1;
    for x in coords_x: 
        for y in coords_y:
            for z in coords_z:
                i += 1;
                instance = FLAMEGPU.agent("ECM").newAgent();
                instance.setVariableInt("id", current_id+i);
                instance.setVariableFloat("x", x);
                instance.setVariableFloat("y", y);
                instance.setVariableFloat("z", z);
                instance.setVariableFloat("vx", 0.0);
                instance.setVariableFloat("vy", 0.0);
                instance.setVariableFloat("vz", 0.0);
                instance.setVariableFloat("fx", 0.0);
                instance.setVariableFloat("fy", 0.0);
                instance.setVariableFloat("fz", 0.0);
                instance.setVariableFloat("k_elast", k_elast);
                instance.setVariableFloat("d_dumping", d_dumping);
                instance.setVariableFloat("mass", mass);
                instance.setVariableFloat("boundary_fx", 0.0);
                instance.setVariableFloat("boundary_fy", 0.0);
                instance.setVariableFloat("boundary_fz", 0.0);
                instance.setVariableFloat("f_extension", 0.0);
                instance.setVariableFloat("f_compression", 0.0);
                instance.setVariableFloat("elastic_energy", 0.0);
                instance.setVariableUInt8("clamped_bx_pos", 0);
                instance.setVariableUInt8("clamped_bx_neg", 0);
                instance.setVariableUInt8("clamped_by_pos", 0);
                instance.setVariableUInt8("clamped_by_neg", 0);
                instance.setVariableUInt8("clamped_bz_pos", 0);
                instance.setVariableUInt8("clamped_bz_neg", 0);

    FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id+i)
    return

# Add function callback to INIT functions for population generation
initialAgentPopulation = initAgentPopulations();
model.addInitFunctionCallback(initialAgentPopulation);
#initialBondaryCornerPopulation = initBoundaryCorners();
#model.addInitFunctionCallback(initialBondaryCornerPopulation);
#initialECMPopulation = initECMPopulation();
#model.addInitFunctionCallback(initialECMPopulation);

"""
  Example step function for later reference
"""
stepCounter = 1
class MoveBoundaries(pyflamegpu.HostFunctionCallback):
     """
     pyflamegpu requires step functions to be a class which extends the StepFunction base class.
     This class must extend the handle function
     """

     # Define Python class 'constructor'
     def __init__(self):
         super().__init__()

     # Override C++ method: virtual void run(FLAMEGPU_HOST_API*);
     def run(self, FLAMEGPU):
         global stepCounter
         global BOUNDARY_COORDS, BOUNDARY_DISP_RATES, TIME_STEP

         if any(dr > 0.0 or dr < 0.0 for dr in BOUNDARY_DISP_RATES):
            
            #coord_boundary = FLAMEGPU.environment.getPropertyArrayFloat("COORDS_BOUNDARIES")
            for i in range(6):                
                BOUNDARY_COORDS[i] += (BOUNDARY_DISP_RATES[i] * TIME_STEP)
            
            bcs = [BOUNDARY_COORDS[0], BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[3], BOUNDARY_COORDS[4], BOUNDARY_COORDS[5]];  #+X,-X,+Y,-Y,+Z,-Z
            FLAMEGPU.environment.setPropertyArrayFloat("COORDS_BOUNDARIES", bcs);
            if DEBUG_PRINTING:
                print ("====== MOVING BOUNDARIES ======"); 
                print ("End of step: ", stepCounter);
                print ("New boundary positions [+X,-X,+Y,-Y,+Z,-Z]: ", BOUNDARY_COORDS);
                print ("==============================="); 
         stepCounter += 1

mb = MoveBoundaries()
model.addStepFunctionCallback(mb)

"""
  Control flow
"""    
# Layer #1
model.newLayer("L1").addAgentFunction("ECM", "ecm_output_location_data");
model.Layer("L1").addAgentFunction("BCORNER", "bcorner_output_location_data");
# Layer #2
model.newLayer("L2").addAgentFunction("ECM", "ecm_boundary_interaction");
# Layer #3
model.newLayer("L3").addAgentFunction("ECM", "ecm_ecm_interaction");
# Layer #4
model.newLayer("L4").addAgentFunction("ECM", "ecm_move");
model.Layer("L4").addAgentFunction("BCORNER", "bcorner_move");


# Create and configure logging details 
logging_config = pyflamegpu.LoggingConfig(model);
logging_config.logEnvironment("CURRENT_ID");
ecm_agent_log = logging_config.agent("ECM");
ecm_agent_log.logCount();
ecm_agent_log.logMeanFloat("fx");
ecm_agent_log.logMeanFloat("fy");
ecm_agent_log.logMeanFloat("fz");
ecm_agent_log.logStandardDevFloat("fx");
ecm_agent_log.logStandardDevFloat("fy");
ecm_agent_log.logStandardDevFloat("fz");
step_log = pyflamegpu.StepLoggingConfig(logging_config);
step_log.setFrequency(1);


"""
  Create Model Runner
"""  
if ENSEMBLE: 
  """
  Create Run Plan Vector
  """   
  run_plan_vector = pyflamegpu.RunPlanVec(model, ENSEMBLE_RUNS);
  run_plan_vector.setSteps(env.getPropertyUInt("STEPS"));
  simulation_seed = random.randint(0,99999);
  run_plan_vector.setRandomSimulationSeed(simulation_seed,1000);
  simulation = pyflamegpu.CUDAEnsemble(model);
else:
  simulation = pyflamegpu.CUDASimulation(model);
  simulation.SimulationConfig().steps = STEPS;
  #if not VISUALISATION:
  # simulation.SimulationConfig().steps = STEPS;

simulation.setStepLog(step_log);
simulation.setExitLog(logging_config)

"""
  Create Visualisation
"""
if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
    visualisation = simulation.getVisualisation();
    # Configure vis
    envWidth = MAX_EXPECTED_BOUNDARY_POS - MIN_EXPECTED_BOUNDARY_POS;
    INIT_CAM = MAX_EXPECTED_BOUNDARY_POS * 4.5;
    # Visualisation.setInitialCameraLocation(INIT_CAM * 2, INIT_CAM, INIT_CAM);
    visualisation.setInitialCameraLocation(0.0, 0.0, INIT_CAM);
    visualisation.setCameraSpeed(0.002 * envWidth);
    circ_ecm_agt = visualisation.addAgent("ECM");    
    # Position vars are named x, y, z; so they are used by default
    circ_ecm_agt.setModel(pyflamegpu.ICOSPHERE);    
    #circ_ecm_agt.setModelScale(env.getPropertyFloat("ECM_ECM_INTERACTION_RADIUS")/7.5);
    circ_ecm_agt.setModelScale(0.06);
    circ_ecm_agt.setColor(pyflamegpu.GREEN);
    #circ_ecm_agt.setColor(pyflamegpu.ViridisInterpolation("y", -1.0, 1.0));
    #circ_ecm_agt.setColor(pyflamegpu.HSVInterpolation("y", 0.0, 360.0));
    # TODO: add variable force to plot colors
    f_max = ECM_K_ELAST * (ECM_ECM_EQUILIBRIUM_DISTANCE);
    max_energy = 0.5 * (f_max * f_max ) / ECM_K_ELAST;
    print("max force, max energy: ", f_max, max_energy);
    circ_ecm_agt.setColor(pyflamegpu.HSVInterpolation.GREENRED("elastic_energy",0.00000001, max_energy * 1.0));
    square_bcorner_agt = visualisation.addAgent("BCORNER");
    square_bcorner_agt.setModel(pyflamegpu.CUBE);
    square_bcorner_agt.setModelScale(0.05);
    square_bcorner_agt.setColor(pyflamegpu.RED);

    pen = visualisation.newLineSketch(1, 1, 1, 0.8); 
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[2], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[2], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[3], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[3], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[3], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[3], BOUNDARY_COORDS[5]);

    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[2], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[3], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[2], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[3], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[3], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[3], BOUNDARY_COORDS[5]);

    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[2], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[3], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[3], BOUNDARY_COORDS[4]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[2], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[2], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[0], BOUNDARY_COORDS[3], BOUNDARY_COORDS[5]);
    pen.addVertex(BOUNDARY_COORDS[1], BOUNDARY_COORDS[3], BOUNDARY_COORDS[5]);

    visualisation.activate();

"""
  Execution
"""
if ENSEMBLE:
    simulation.simulate(run_plan_vector);
else:
    simulation.simulate();

"""
  Export Pop
"""
# simulation.exportData("end.xml");

# Join Visualisation
if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
    visualisation.join();



# Deal with logs
if ENSEMBLE:
    logs = simulation.getLogs();
else:
    logs = simulation.getRunLog();


if ENSEMBLE:
    agent_counts = [None]*ENSEMBLE_RUNS   
    current_id = [None]*ENSEMBLE_RUNS
    # Read logs
    for i in range(len(logs)):
      sl = logs[i].getStepLog();
      agent_counts[i] = ["ensemble_run_"+str(i)+"_ecm"];
      ecm_force_mean[i] = ["ensemble_run_"+str(i)];
      ecm_force_std[i] = ["ensemble_run_"+str(i)];
      current_id[i] = ["ensemble_run_"+str(i)];
      for step in sl:
        # Collect step data
        ecm_agents = step.getAgent("ECM");        
        current_id[i].append(step.getEnvironmentPropertyUInt("CURRENT_ID"));
        # Collect agent data from step
        ecm_force_mean[i][0].append(ecm_agents.getMean("fx"));
        ecm_force_mean[i][1].append(ecm_agents.getMean("fy"));
        ecm_force_mean[i][2].append(ecm_agents.getMean("fz"));
        ecm_force_std[i][0].append(ecm_agents.getStandardDev("fx"));
        ecm_force_std[i][1].append(ecm_agents.getStandardDev("fy"));
        ecm_force_std[i][2].append(ecm_agents.getStandardDev("fz"));
        agent_counts[i][0].append(ecm_agents.getCount());

    print()


    #Print warning data
    print("Agent counts per step per ensemble run")
    for j in range(len(agent_counts)):
      for k in range(len(agent_counts[j])):
        print(agent_counts[j][k])
    print()
   
    """
      Boid graph generation for future reference
    """
    # Generate graphs 
    # for j in range(ENSEMBLE_RUNS):
    #     # Plot 3d graph of average flock position over simulation for individual model run
    #     fig = plt.figure(figsize=(8,8));
    #     ax = fig.gca(projection='3d');
    #     ax.set_xlabel("Model environment x");
    #     ax.set_ylabel("Model environment y");
    #     ax.set_zlabel("Model environment z");
    #     fig.suptitle("Ensemble run "+str(j)+" boids mean flock positions",fontsize=16);
    #     label = "Boids mean flock position, ensemble run "+str(j);
    #     fname = "average_flock_positions_run"+str(j)+".png";
    #     # Position start and finish flock position text
    #     for k in text_pos[j]:
    #         ax.text(k[0],k[1],k[2],k[3],None);
    #     ax.plot(positions_mean[j][0], positions_mean[j][1], positions_mean[j][2], label=label);
    #     ax.set_xlim3d([-1.0,1.0]);
    #     ax.set_ylim3d([-1.0,1.0]);
    #     ax.set_zlim3d([-1.0,1.0]);
    #     ax.legend();
    #     plt.savefig(fname,format='png');
    #     plt.close(fig);

    #     # Plot graphs for average of each fx, fy, and fz with standard deviation error bars
    #     steplist = range(STEPS);
    #     fig,(axx,axy,axz) = plt.subplots(1,3, figsize=(21,6));
    #     fig.suptitle("Ensemble run "+str(j)+" mean boid velocities with std errorbar",fontsize=16);
    #     velfname = "mean_velocities_run"+str(j)+".png";
    #     axx.errorbar(steplist,velocities_mean[j][0],yerr=velocities_std[j][0],elinewidth=0.5,capsize=1.0);
    #     axx.set_xlabel("Simulation step");
    #     axx.set_ylabel("Boid agents average fx");
    #     axy.errorbar(steplist,velocities_mean[j][1],yerr=velocities_std[j][1],elinewidth=0.5,capsize=1.0);
    #     axy.set_xlabel("Simulation step");
    #     axy.set_ylabel("Boid agents average fy");
    #     axz.errorbar(steplist,velocities_mean[j][2],yerr=velocities_std[j][2],elinewidth=0.5,capsize=1.0);
    #     axz.set_xlabel("Simulation step");
    #     axz.set_ylabel("Boid agents average fz");
    #     plt.savefig(velfname,format='png');
    #     plt.close(fig);

    # # Plot every model in esemble's average flock position over simulation on the same 3d graph
    # fig = plt.figure(figsize=(12,12));
    # fig.suptitle("Ensemble Boids mean flock positions",fontsize=16);
    # ax = fig.gca(projection='3d');
    # ax.set_xlabel("Model environment x");
    # ax.set_ylabel("Model environment y");
    # ax.set_zlabel("Model environment z");
    # fname = "ensemble_average_flock_positions.png";
    # ## Plot start and finish text for each flock path ---VERY CLUTTERED---
    # # for i in text_pos:
    # #     for k in i:
    # #         ax.text(k[0],k[1],k[2],k[3],'x');
    # jcount = 0;
    # for j in positions_mean:
    #     label1 = "Run "+str(jcount);
    #     ax.plot(j[0], j[1], j[2], label=label1);
    #     jcount+=1;
    # #ax.set_xlim3d([-1.0,1.0]);
    # #ax.set_ylim3d([-1.0,1.0]);
    # #ax.set_zlim3d([-1.0,1.0]);
    # ax.legend();
    # plt.savefig(fname,format='png');
    # plt.close(fig);
else:
    steps = logs.getStepLog();
    ecm_agent_counts = [None]*len(steps)
    counter = 0;
    for step in steps:
        stepcount = step.getStepCount();
        ecm_agents = step.getAgent("ECM");
        ecm_agent_counts[counter] = ecm_agents.getCount();
        counter+=1;
    print()
    print("Agent counts per step")
    #for j in range(len(steps)):
    #  print("step",j,"ECM",ecm_agent_counts[j]) 