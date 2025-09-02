def my_split(sentence, separator):
    result = []
    word = ''
    for char in sentence:
        if char == separator:
            result.append(word)
            word = ''
        else:
            word += char
    result.append(word)  # Add the last word
    return result

def my_join(items, separator):
    result = ''
    for i in range(len(items)):
        result += items[i]
        if i < len(items) - 1:
            result += separator
    return result

# Main program
sentence = input("Please enter sentence:")
words = my_split(sentence, ' ')
print(my_join(words, ','))
for word in words:
    print(word)