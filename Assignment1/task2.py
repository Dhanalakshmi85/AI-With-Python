def main():
    grocery_list = []

    while True:
        print("1. Add\n2. Remove\n3. Quit")
        choice = input("Choose an action: ")

        if choice == "1":
            item = input("What will be added?: ")
            grocery_list.append(item)

        elif choice == "2":
            if not grocery_list:
                print("The list is empty.")
                continue

            print(f"There are {len(grocery_list)} items in the list.")
            try:
                index = int(input("Which item is deleted?: "))
                if 0 <= index < len(grocery_list):
                    del grocery_list[index]
                else:
                    print("Incorrect selection.")
            except ValueError:
                print("Incorrect selection.")

        elif choice == "3":
            print("The following items remain in the list:")
            for item in grocery_list:
                print(item)
            break

        else:
            print("Incorrect selection.")
