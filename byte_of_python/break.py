#!usr/bin/python3
# Filename: break.py

while True: # Fakes R's repeat loop
  s = (input('Enter something: '))
  if s == 'quit': # Provide condition to end the loop
    break
  print('Length of the string is ', len(s))
print('Done')