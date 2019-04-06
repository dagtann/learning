#!/usr/bin/python3
# Filename: if.py

number = 23
guess = int(input('Enter an integer : '))
# save the user's guess from the input function
# input() saves strings, hence, guess is reclassed to int
# using int()

if guess == number:
  # Indentation shows what statements belong to what block.
  print('Congratulations, you guessed it.')
  print('(but you do not win any prizes!)')
elif guess < number:
  # abbreviates another if else statement
  print('No, it is a little higher than that')
else:
  print('No, it is a little lower than that.')
  # use of elif and else is entirely optional


print('Done')
# This last statement is always executed, after the if
# statement is executed.