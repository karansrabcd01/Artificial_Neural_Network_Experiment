function basic_operations()
    x = input('Enter first number: ');
    y = input('Enter second number: ');
    fprintf('Addition: %f\n', add(x, y));
    fprintf('Subtraction: %f\n', subtract(x, y));
    fprintf('Multiplication: %f\n', multiply(x, y));
    fprintf('Division: %s\n', divide(x, y));
end

function result = add(a, b)
    result = a + b;
end

function result = subtract(a, b)
    result = a - b;
end
 
function result = multiply(a, b)
    result = a * b;
end

function result = divide(a, b)
    if b ~= 0
        result = num2str(a / b);
    else
        result = 'Error! Division by zero.';
    end
end
