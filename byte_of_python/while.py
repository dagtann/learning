#!usr/bin/python
# Filename: while.py

number = 23
running = True

while running:
  guess = int(input('Enter an integer: '))

  if guess == number:
    print('Congratulations, you guessed it.')
    running = False # Stop loop
  elif guess < number:
    print('No, it is a little higher than that.')
  else:
    print('No it is a little lower than that.')
else: # Python allows else statements for while loops
  print('The while loop has ended.')

print('Done')