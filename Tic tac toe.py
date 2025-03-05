def minimax(board, depth, is_maximizing):
    if check_win(board, 'X'):  # AI is 'X'
        return 1
    if check_win(board, 'O'):
        return -1
    if check_draw(board):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ''
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ''
                    best_score = min(score, best_score)
        return best_score


def find_best_move(board):
    best_score = -float('inf')
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == '':
                board[i][j] = 'X'
                score = minimax(board, 0, False)
                board[i][j] = ''
                if score > best_score:
                    best_score = score
                    best_move = (i, j)

    return best_move


def check_win(board, player):
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if board[0][0] == board[1][1] == board[2][2] == player or board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False


def check_draw(board):
    return all(board[i][j] != '' for i in range(3) for j in range(3))


def print_board(board):
    for row in board:
        print('|'.join(cell if cell else ' ' for cell in row))
    print('-' * 5)


board = [['', '', ''], ['', '', ''], ['', '', '']]

while True:
    print_board(board)
    row, col = map(int, input("Enter your move (row and column 0-2): ").split())
    if board[row][col] == '':
        board[row][col] = 'O'
    else:
        print("Invalid move. Try again.")
        continue

    if check_win(board, 'O'):
        print_board(board)
        print("You win!")
        break
    elif check_draw(board):
        print_board(board)
        print("It's a draw!")
        break

    ai_move = find_best_move(board)
    if ai_move:
        board[ai_move[0]][ai_move[1]] = 'X'
    else:
        print("No valid moves left!")
        break

    if check_win(board, 'X'):
        print_board(board)
        print("AI wins!")
        break
    elif check_draw(board):
        print_board(board)
        print("It's a draw!")
        break