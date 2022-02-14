from ast import Mod
from enum import Enum
import random
import uuid
import networkx as nx
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

def perceived_proposal_value(model):
    """Returns the community's perceived value of a proposal at each time step

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """
#     values = [a.current_proposal_evaluation for a in model.grid.get_all_cell_contents()]
    return model.perceived_proposal_value

def true_proposal_value(model):
    """Returns the true value of the proposal being considered by the community.

    The value of this proposal is theoretical. It represents that actual value, free from any bias
    that will be delivered to the organization if it passes.

    Agents only perceive the value of this proposal through the lens of their bias.
    There are a set number of days to build consensus through their network.
    Each day a random number of each agent's close connections meet and share their view on how valuable a proposal is.
    At the end of the day the agent and their connections update their perception of the proposal value
    to the be the average of everyone's valuation.


    Args:
        model (Mesa.Model.MolochDAO): a Mesa MolochDAO object being simulated.
    """
    return model.true_proposal_value

def realized_proposal_value(model):
    """Returns the cumulative value of proposals that were passed by the org.

    Args:
        model ([type]): [description]
    """
    return model.realized_proposal_value

def generate_proposal(dimensions: int) -> dict:
    id = uuid.uuid4()
    values = [random.uniform(0, 1) for _ in range(dimensions)]
    return values

class MolochDAO(Model):
    def __init__(
        self, 
        num_nodes = 3,
        avg_node_degree = 2,
        proposal_dimension = 2, # number of categories considered in evaluating the value of the proposal
        evaluation_period = 3, # num. time steps for agents to evaluate the proposal
        num_proposals = 3
    ) -> None:

        self.num_nodes = num_nodes
        prob = avg_node_degree / num_nodes
        self.G = nx.watts_strogatz_graph(n = self.num_nodes, k = avg_node_degree, p = prob)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.proposal_dimension = proposal_dimension
        self.evaluation_period = evaluation_period
        # data variables to track
        self.true_proposal_value = 0
        self.realized_proposal_value = 0 # initialize the cumulative value of proposals passed by the org.
        self.perceived_proposal_value = 0
        # self.proposals = [random.uniform(0,1) for _ in range(num_proposals)]
        # self.num_outstanding_proposals = num_proposals
        self.datacollector = DataCollector(
            {
                # "perceived_proposal_value": perceived_proposal_value(self)
                "perceived_proposal_value": perceived_proposal_value,
                "true_proposal_value": true_proposal_value,
                "realized_proposal_value": realized_proposal_value
            }
        )

        # Create agents on the network
        for i, node in enumerate(self.G.nodes()):
            # generate random agent bias
            agent_bias = [random.uniform(0, 2) for i in range(self.proposal_dimension)]
            # set the support threshold
            ## option 1: if perceived value is greater than actual (unknown to agent) max value.
            ## 
            support_threshold = random.uniform(0, self.proposal_dimension) # between 0 (will support anything) and max perceived value of the proposal (it must be perfect)
            
            # create agent
            agent = MemberAgent(
                unique_id=i,
                model=self,
                bias=agent_bias,
                support_threshold = support_threshold
            )
            self.schedule.add(agent)
            # add agent to node on the network
            self.grid.place_agent(agent, node)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run(self) -> None:
        # while self.num_outstanding_proposals > 0:
        #     self.step()

        # list of all agents in the model
        agents = self.grid.get_cell_list_contents(self.G.nodes())

        # print(self.G.nodes())

        # each iteration of the model (currently set to 5)
        for i in range(15):
            # create a new proposal
            proposal = generate_proposal(self.proposal_dimension)
            votes = [] # initialize empty vote bank
            perceived_value = [] # initialize empty perceived value tracker
            # record (theoretical) true proposal value
            self.true_proposal_value += sum(proposal)
            
            
            # all agents evaluate the new proposal
            for agent in agents:
                agent.evaluate_proposal(proposal)

            # influence neighbors for specified number of steps (currently set to 3)
            for i in range(3):
                self.step() # this activates each agent's step function

            # vote on proposal then reset evaluation
            for agent in agents:
                vote = agent.vote()
                # if more than 50% vote add proposal value to org.
                if vote:
                    votes.append(1)
                else:
                    votes.append(0)
                # print(vote)
                # agent's reset their proposal opinion and get ready for the next one
                agent.reset_evaluation()
            
            # if > 50% of vote is made
            if sum(votes) / self.num_nodes > 0.5:
                self.realized_proposal_value += sum(proposal)

class MemberAgent(Agent):
    def __init__(
        self, 
        unique_id: int, 
        model: Model,
        bias: list,
        support_threshold: float
    ) -> None:
        super().__init__(unique_id, model)

        self.bias = bias
        self.support_threshold = support_threshold
        self.current_proposal_evaluation = 0

    def evaluate_proposal(self, proposal: list) -> None:
        """Evaluate the value of a proposal by applying personal bias.

        Args:
            proposal ([list]): the true value of the proposal on each dimension upon which its considered.

        Returns:
            float: agent's perception of value of the proposal after applying bias
        """
        perceived_value = sum([a*b for a,b in zip(self.bias, proposal)])
        self.current_proposal_evaluation = perceived_value
    
    def influence_neighbors(self):
        # print("starting opinion", self.current_proposal_evaluation)
        # connect with random number of neighbors and influence their perception
        neighbor_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)

        # talk to a random number of friends and update perception of proposal value to the average opinion

        k = random.randint(1, len(neighbor_nodes)) # between 1 and all connected nodes
        random_friends = self.random.sample(neighbor_nodes, k)
        
        # get friends evaluations of the current proposal
        friends_evaluation = [a.current_proposal_evaluation for a in self.model.grid.get_cell_list_contents(random_friends)]

        # update personal opinion to the average of friends opinions who the agent talks to
        self.current_proposal_evaluation = (sum(friends_evaluation) + self.current_proposal_evaluation) / (1 + k)

        # print("opinion after consensus", self.current_proposal_evaluation)

    def vote(self):
        support = self.support_threshold < self.current_proposal_evaluation
        return support
    
    def reset_evaluation(self) -> None:
        self.current_proposal_evaluation = 0
    
    def step(self):
        """agent activity that occurs each time step.
        """
        # evaluate a new proposal
        # proposal = generate_proposal(3)

        # self.evaluate_proposal(proposal)

        # build consensus by influencing neighbors for defined number of steps
        self.influence_neighbors()

        # vote on proposal

if __name__ == "__main__":
    model = MolochDAO()
    model.run()
    model_df = model.datacollector.get_model_vars_dataframe()
    model_df.head()
    model_df.plot()
        
