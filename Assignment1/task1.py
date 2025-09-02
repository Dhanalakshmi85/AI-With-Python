def tester(givenstring="Too short"):
    print(givenstring)

def main():
    while True:
        user_input = input("Write something (quit ends): ")
        if user_input == "quit":
            break
        if len(user_input) < 10:
            tester()  # uses default value
        else:
            tester(user_input)

# Start the program
main()