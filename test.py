user_code = input("Enter code to execute: ")
# Vulnerable: Evaluating user input as code
eval(user_code) 