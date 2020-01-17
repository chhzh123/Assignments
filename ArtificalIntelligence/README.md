## Artificial Intelligence

This is the repository of chhzh123's assignments of *Artificial Intelligence* - Fall 2019 @ SYSU lectured by *Yongmei Liu*.

There are 16 [labs](#labs), 4 [projects](#projects), and 4 [theory assignments](#theory-assignments) in this course. Brief introduction is listed below, and detailed descriptions can be found in each folder.

The assignments are primarily coded in Python, expect for some listed below use other languages.
* E03 uses C++
* E05, E06 use [Prolog](https://www.swi-prolog.org/)
* E07, E08 use PDDL Planner, please refer to the following links
    * Online PDDL Planner, <http://editor.planning.domains/>
    * Metric-FF Planer, <https://fai.cs.uni-saarland.de/hoffmann/metric-ff.html>

### Labs
Lab assignments have the prefix `E`.

#### E01 - Maze
Use **BFS** or **DFS** to solve the maze problem (i.e., find the shortest path from the start point `S` to the ending point `E`).

```
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 S%
% %%%%%%%%%%%%%%%%%%%%%%% %%%%%%%% %
% %%   %   %      %%%%%%%   %%     %
% %% % % % % %%%% %%%%%%%%% %% %%%%%
% %% % % % %             %% %%     %
% %% % % % % % %%%%  %%%    %%%%%% %
% %  % % %   %    %% %%%%%%%%      %
% %% % % %%%%%%%% %%        %% %%%%%
% %% %   %%       %%%%%%%%% %%     %
%    %%%%%% %%%%%%%      %% %%%%%% %
%%%%%%      %       %%%% %% %      %
%      %%%%%% %%%%% %    %% %% %%%%%
% %%%%%%      %       %%%%% %%     %
%        %%%%%% %%%%%%%%%%% %%  %% %
%%%%%%%%%%                  %%%%%% %
%E         %%%%%%%%%%%%%%%%        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

#### E02 - 15 Puzzle
Use **IDA\*** to solve the 15-Puzzle problem.

![15 puzzle](E02_15Puzzle/fig/case1.png)


#### E03 - Othello
Use **minimax** and **Alpha-Beta pruning** to implement the [Othello](http://www.tothello.com/index.html) game.

![othello](E03_Othello/fig/othello.png)

#### E04 - Futoshiki
Use **forward checking (FC)** algorithm to solve the [Futoshiki](http://www.futoshiki.org/) puzzle.

![futoshiki](E04_Futoshiki/fig/futoshiki1.png)

#### E05 - Family
Use **Prolog** to describe the family relationship, and answer the quiries about some relationship, e.g. m-th cousin n times removed.

![family tree](E05_Family/fig/family.png)

#### E06 - Queries on Knowledge Base
Given a Knowledge Base describing the distribution of branches of 10 well-known restaurants in Guangzhou. Use **Prolog** to answer the queries like

```
What districts have restaurants of yuecai and xiangcai?
What areas have two or more restaurants?
Which restaurant has the longest history?
```

#### E07 - FF Planer
Define the domains and problems of *8-puzzle* and *blockworlds* using **PDDL**, and use **FF Planer** to plan a schedule.

#### E08 - Boxman
Transform the boxman game into a planning problem, and use **FF Planer** to obtain the movements of the boxman.

![boxman](E08_Boxman/fig/case4.png)

#### E09 - Bayesian Network
Build Bayesian networks of two problems (Bureglary & Diagnosing) using Python package **Pomegranate** and do the inference.

![bayesian network](E09_BN/fig/burglary.png)

#### E10 - Variable Elimination
Implement the **Variable Elimination (VE)** algorithm for Bayesian network and solve the above Bureglary problem.

#### E11 - Decision Tree
Implement the **ID3** decision tree. (Notice: Not allowed to use existing ML packages like `sklearn`.)

Use the [Adult Dataset](http://archive.ics.uci.edu/ml/datasets/Adult) to predict whether a person makes over 50K a year.

#### E12 - Naive Bayes
Implement the **Naive Bayes** algorithm.

Again, use the [Adult Dataset](http://archive.ics.uci.edu/ml/datasets/Adult) to make a prediction.

#### E13 - Expectation-Maximization (EM)
Implement the **EM algorithm** and classify the given 16 footbool teams into 3 classes according to their performance.

#### E14 - Backpropagation
Implement the **backpropagation** algorithm for neural network.

Use the [horse colic dataset](http://archive.ics.uci.edu/ml/datasets/Horse+Colic) to predict whether a horse with colic will live or die.

#### E15 - Reinforcement Learning
Implement the **Q-Learning** algorithm and train the model for the [flappy bird](http://flappybird.io/) game.

![flappy bird](E15_RL/fig/bird.jpg)

#### E16 - Deep Learning
Implement a three-layer **Convolutional Neural Network (CNN)** using the [CS231n framework](http://cs231n.github.io/assignments2019/assignment2/) and do the classification on [CIFAR_10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html). (Notice: Not allowed to use DL frameworks like `tensorflow` and `pytorch`.)

### Projects

### Theory Assignments