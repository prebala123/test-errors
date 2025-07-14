# Code that executes a user's mathematical function
# Example Input: '1 + 1'
# Output: '2'
user_code = input("Enter math expression to execute: ")
# Vulnerable: Evaluating user input as code
eval(user_code) 
