# Reinforcement Learning for the Snake Game

Some experiments of using reinforcement learning to solve the classic snake game. What is special about reinforcement learning is that you don't need to teach the program any tricks about the game, but the program can learn how to play while playing the game automatically.

A deep neural network is used to fit the Q function in Q learning. The inputs are the snake and the fruit's locations, and the outputs are the Q values for 4 actions (UP / DOWN / LEFT / RIGHT). 

I do not have a GPU, so I only tried some small size of game board (5\*5, 7\*7, 10\*10). In theory, this should work in any size, but a larger size may need more training and playing (for the program to see more states of the game and learn).

## Files

"main_train.py" is the file to run. It will use "snake_game.py", which is the simulation of the snake game. "train{m}_{n}.py" stores the hyper-parameters and the model structure for the {m}\*{n} size of game board. Each size's experiment result is stored in a corresponding folder. Each folder has a Jupyter notebook to see the score results.

"make_gif.py" is the codes to make the gif images.

## Results

### Videos 

|Board Size      | 5\*5                       | 7\*7                        | 10\*10                        |
| -------------- |: -------------------------:|:---------------------------:| -----------------------------:|
|Video           | ![Play GIF](./5_5/play.gif)| ![Play GIF](./7_7/play.gif) | ![Play GIF](./10_10/play.gif) |
|Games played    |                            |                             |                               |

### Score plots

(MA(50) = Moving average 50)

(Also see *analyze_all.ipynb*)
![Score plot](./5_5/plot50.png)
![Score plot](./7_7/plot50.png)
![Score plot](./10_10/plot50.png)

Comparison:
![Score plot](./plot50.png)


(To be continued...)
