# Problem Set 2, hangman.py
# Name: Chen Hongzheng chenhzh37@mail2.sysu.edu.cn

# Hangman Game
# -----------------------------------
# Helper code
# You don't need to understand this helper code,
# but you will have to know how to use the functions
# (so be sure to read the docstrings!)
import random
import string

WORDLIST_FILENAME = "words.txt"


def load_words():
    """
    Returns a list of valid words. Words are strings of lowercase letters.
    
    Depending on the size of the word list, this function may
    take a while to finish.
    """
    print("Loading word list from file...")
    # inFile: file
    inFile = open(WORDLIST_FILENAME, 'r')
    # line: string
    line = inFile.readline()
    # wordlist: list of strings
    wordlist = line.split()
    print("  ", len(wordlist), "words loaded.")
    return wordlist



def choose_word(wordlist):
    """
    wordlist (list): list of words (strings)
    
    Returns a word from wordlist at random
    """
    return random.choice(wordlist)

# end of helper code

# -----------------------------------

# Load the list of words into the variable wordlist
# so that it can be accessed from anywhere in the program
wordlist = load_words()


def is_word_guessed(secret_word, letters_guessed):
    '''
    secret_word: string, the word the user is guessing; assumes all letters are
      lowercase
    letters_guessed: list (of letters), which letters have been guessed so far;
      assumes that all letters are lowercase
    returns: boolean, True if all the letters of secret_word are in letters_guessed;
      False otherwise
    '''
    for c in secret_word:
        if not (c in letters_guessed):
            return False
    return True



def get_guessed_word(secret_word, letters_guessed):
    '''
    secret_word: string, the word the user is guessing
    letters_guessed: list (of letters), which letters have been guessed so far
    returns: string, comprised of letters, underscores (_), and spaces that represents
      which letters in secret_word have been guessed so far.
    '''
    res = ""
    for c in secret_word:
        if not (c in letters_guessed):
            res += "_"
        else:
            res += c
    return res



def get_available_letters(letters_guessed):
    '''
    letters_guessed: list (of letters), which letters have been guessed so far
    returns: string (of letters), comprised of letters that represents which letters have not
      yet been guessed.
    '''
    res = string.ascii_lowercase
    for c in letters_guessed:
        index = res.find(c)
        res = res[:index] + res[index+1:]
    return res
    
    

def hangman(secret_word):
    '''
    secret_word: string, the secret word to guess.
    
    Starts up an interactive game of Hangman.
    
    * At the start of the game, let the user know how many 
      letters the secret_word contains and how many guesses s/he starts with.
      
    * The user should start with 6 guesses

    * Before each round, you should display to the user how many guesses
      s/he has left and the letters that the user has not yet guessed.
    
    * Ask the user to supply one guess per round. Remember to make
      sure that the user puts in a letter!
    
    * The user should receive feedback immediately after each guess 
      about whether their guess appears in the computer's word.

    * After each guess, you should display to the user the 
      partially guessed word so far.
    
    Follows the other limitations detailed in the problem write-up.
    '''
    print("Welcome to the game Hangman!")
    print("I am thinking of a word that is {} letters long.".format(len(secret_word)))
    cnt_all = 6
    cnt = 6
    warning = 3
    guessed = []
    print("You have {} warnings left.".format(warning))
    while (cnt > 0):
        print("-------------")
        print("You have {} guesses left.".format(cnt))
        print("Available Letters: " + get_available_letters(guessed))
        c = input("Please guess a letter: ")
        if not c.isalpha():
            if warning > 0:
                warning -= 1
                print("Oops! That is not a valid letter. You have {} warnings left: ".format(warning) + get_guessed_word(secret_word,guessed))
            else:
                cnt -= 1
                print("Oops! That is not a valid letter. You have no warnings left\nso you lose one guess: " + get_guessed_word(secret_word,guessed))
            continue
        c = c.lower()
        if c in guessed:
            if warning > 0:
                warning -= 1
                print("Oops! You've already guessed that letter. You now have {} warnings: ".format(warning) + get_guessed_word(secret_word,guessed))
            else:
                cnt -= 1
                print("Oops! You've already guessed that letter. You have no warnings left\nso you lose one guess: " + get_guessed_word(secret_word,guessed))
            continue
        guessed += c
        if (c in secret_word):
            print("Good guess: " + get_guessed_word(secret_word,guessed))
            if is_word_guessed(secret_word,guessed):
                print("-------------")
                print("Congratulations, you won!")
                print("Your total score for this game is: {}".format((cnt_all-cnt)*len(set(secret_word))))
                return
        else:
            if c in ['a','e','i','o','u']:
                cnt -= 2
            else:
                cnt -= 1
            print("Oops! That letter is not in my word: " + get_guessed_word(secret_word,guessed))
    print("-------------")
    print("Sorry, you ran out of guesses. The word was {}.".format(secret_word))



# When you've completed your hangman function, scroll down to the bottom
# of the file and uncomment the first two lines to test
#(hint: you might want to pick your own
# secret_word while you're doing your own testing)


# -----------------------------------



def match_with_gaps(my_word, other_word):
    '''
    my_word: string with _ characters, current guess of secret word
    other_word: string, regular English word
    returns: boolean, True if all the actual letters of my_word match the 
        corresponding letters of other_word, or the letter is the special symbol
        _ , and my_word and other_word are of the same length;
        False otherwise: 
    '''
    no_space_word = my_word.strip()
    if (len(no_space_word) != len(other_word)):
        return False
    for i in range(len(no_space_word)):
        if not (my_word[i] == '_' or my_word[i] == other_word[i]):
            return False
    return True



def show_possible_matches(my_word):
    '''
    my_word: string with _ characters, current guess of secret word
    returns: nothing, but should print out every word in wordlist that matches my_word
             Keep in mind that in hangman when a letter is guessed, all the positions
             at which that letter occurs in the secret word are revealed.
             Therefore, the hidden letter(_ ) cannot be one of the letters in the word
             that has already been revealed.

    '''
    flag = False
    for word in wordlist:
        if match_with_gaps(my_word,word):
            flag = True
            print(word,end=" ")
    if not flag:
        print("No matches found")
    else:
        print()



def hangman_with_hints(secret_word):
    '''
    secret_word: string, the secret word to guess.
    
    Starts up an interactive game of Hangman.
    
    * At the start of the game, let the user know how many 
      letters the secret_word contains and how many guesses s/he starts with.
      
    * The user should start with 6 guesses
    
    * Before each round, you should display to the user how many guesses
      s/he has left and the letters that the user has not yet guessed.
    
    * Ask the user to supply one guess per round. Make sure to check that the user guesses a letter
      
    * The user should receive feedback immediately after each guess 
      about whether their guess appears in the computer's word.

    * After each guess, you should display to the user the 
      partially guessed word so far.
      
    * If the guess is the symbol *, print out all words in wordlist that
      matches the current guessed word. 
    
    Follows the other limitations detailed in the problem write-up.
    '''
    print("Welcome to the game Hangman!")
    print("I am thinking of a word that is {} letters long.".format(len(secret_word)))
    cnt = 6
    warning = 3
    guessed = []
    print("You have {} warnings left.".format(warning))
    while (cnt > 0):
        print("-------------")
        print("You have {} guesses left.".format(cnt))
        print("Available Letters: " + get_available_letters(guessed))
        c = input("Please guess a letter: ")
        # New for Part 3
        if c == '*':
            print("Possible word matches are:")
            show_possible_matches(get_guessed_word(secret_word,guessed))
            continue
        ######
        if not c.isalpha():
            if warning > 0:
                warning -= 1
                print("Oops! That is not a valid letter. You have {} warnings left: ".format(warning) + get_guessed_word(secret_word,guessed))
            else:
                cnt -= 1
                print("Oops! That is not a valid letter. You have no warnings left\nso you lose one guess: " + get_guessed_word(secret_word,guessed))
            continue
        c = c.lower()
        if c in guessed:
            if warning > 0:
                warning -= 1
                print("Oops! You've already guessed that letter. You now have {} warnings: ".format(warning) + get_guessed_word(secret_word,guessed))
            else:
                cnt -= 1
                print("Oops! You've already guessed that letter. You have no warnings left\nso you lose one guess: " + get_guessed_word(secret_word,guessed))
            continue
        guessed += c
        if (c in secret_word):
            print("Good guess: " + get_guessed_word(secret_word,guessed))
            if is_word_guessed(secret_word,guessed):
                print("-------------")
                print("Congratulations, you won!")
                print("Your total score for this game is: {}".format(cnt*len(set(secret_word))))
                return
        else:
            if c in ['a','e','i','o','u']:
                cnt -= 2
            else:
                cnt -= 1
            print("Oops! That letter is not in my word: " + get_guessed_word(secret_word,guessed))
    print("-------------")
    print("Sorry, you ran out of guesses. The word was {}.".format(secret_word))



# When you've completed your hangman_with_hint function, comment the two similar
# lines above that were used to run the hangman function, and then uncomment
# these two lines and run this file to test!
# Hint: You might want to pick your own secret_word while you're testing.


if __name__ == "__main__":

    # To test part 2, comment out the pass line above and
    # uncomment the following two lines.
    
    # secret_word = choose_word(wordlist)
    # hangman(secret_word)

###############
    
    # To test part 3 re-comment out the above lines and 
    # uncomment the following two lines. 
    
    secret_word = choose_word(wordlist)
    hangman_with_hints(secret_word)