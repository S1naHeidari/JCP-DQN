# Jack's Car Rental Problem

## Problem Description

Jack manages two locations for a nationwide car rental company. Each day, customers arrive at each location to rent cars. If Jack has cars available, he rents them out and is credited $10 by the national company. If he is out of cars at a location, then the business is lost. Cars become available for renting the day after they are returned. To ensure cars are available where needed, Jack can move them between the two locations overnight, at a cost of $2 per car moved.

We assume the number of cars requested and returned at each location follows Poisson distributions. Suppose the expected rental requests and returns are 3 and 4 for the first location, and 3 and 2 for the second location. To simplify, each location can have no more than 20 cars, and a maximum of five cars can be moved between locations in one night. The discount rate is set to 0.9.

This problem can be formulated as a continuing finite Markov Decision Process (MDP), where time steps are days, states represent the number of cars at each location at the end of the day, and actions represent the net number of cars moved between locations overnight.

## Policies and Solution

Policy iteration is applied to find the optimal policies for this problem. The process starts from a policy that doesn't move any cars and iteratively improves the policies. Figure 4.2 in the source material shows the sequence of policies found through policy iteration.

## How to Use

This example can serve as a basis for solving similar problems in reinforcement learning and dynamic programming. The provided problem description, assumptions, and formulation can be adapted to various scenarios where decision-making is involved.

To understand the step-by-step process and see the detailed implementation, refer to the source material and relevant resources.

## Source

This README is based on Example 4.2 from the book "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto. The example illustrates a car rental problem and its formulation as a continuing finite MDP. For more details, refer to the source material.

---

*Note: This README is a summary and interpretation of the original text from the mentioned source. For accurate and detailed information, please refer to the source material.*