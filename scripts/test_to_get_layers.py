
def get_layers_number(number):
    # 16 -> 256
    # 32 -> 128
    # 64 -> 64
    numbers = [16, 32, 64, 128, 256]
    numb_position = numbers.index(number) + 1
    return numbers[-numb_position]

def get_layers_number(number):
    # 64 -> 256
    # 32 -> 128
    numbers = [16, 32, 64, 128, 256]
    current_position = numbers.index(number)
    numb_position = numbers.index(number) + 1
    return numbers[(current_position*2) -1]

print(get_layers_number(16))
print(get_layers_number(32))
print(get_layers_number(64))
