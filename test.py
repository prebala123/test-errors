# Code that safely executes a user's mathematical expression
import ast
import operator

def safe_eval(expr):
    # Define allowed operations
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg
    }

    def _eval(node):
        # Numerical literal
        if isinstance(node, ast.Num):
            return node.n
        # Unary operations like negation
        elif isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in allowed_ops:
                raise ValueError(f"Unsupported unary operation: {op}")
            return allowed_ops[op](_eval(node.operand))
        # Binary operations like addition, subtraction
        elif isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in allowed_ops:
                raise ValueError(f"Unsupported binary operation: {op}")
            return allowed_ops[op](_eval(node.left), _eval(node.right))
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    # Parse the input expression
    try:
        parsed = ast.parse(expr, mode='eval')
        return _eval(parsed.body)
    except SyntaxError:
        print("Invalid mathematical expression.")
        return None

# Prompt for input and safely evaluate
user_code = input("Enter math expression to execute: ")
result = safe_eval(user_code)
if result is not None:
    print(f"Result: {result}")