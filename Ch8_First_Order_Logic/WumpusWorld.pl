%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wumpus World (4x4 Grid) Knowledge Base in Prolog
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BOARD DEFINITION (4x4)
% Valid coordinates: (X,Y) where X,Y ∈ {1,2,3,4}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

coord(1). coord(2). coord(3). coord(4).

square((X,Y)) :- coord(X), coord(Y).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADJACENCY RULE
% Squares are adjacent if they differ by 1 in either X or Y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

adjacent((X,Y), (X1,Y)) :-
    coord(X), coord(Y),
    X1 is X + 1,
    coord(X1).

adjacent((X,Y), (X1,Y)) :-
    coord(X), coord(Y),
    X1 is X - 1,
    coord(X1).

adjacent((X,Y), (X,Y1)) :-
    coord(X), coord(Y),
    Y1 is Y + 1,
    coord(Y1).

adjacent((X,Y), (X,Y1)) :-
    coord(X), coord(Y),
    Y1 is Y - 1,
    coord(Y1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WORLD CONFIGURATION (EDIT THIS SECTION)
% Define where pits, wumpus, and gold are located
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example configuration (you can change these)

wumpus((3,1)).
pit((2,3)).
pit((4,4)).
gold((3,3)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GAME RULES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A square has a stench if it is adjacent to a square with a wumpus
stench((X,Y)) :-
    adjacent((X,Y), (X1,Y1)),
    wumpus((X1,Y1)).

% A square has a breeze if it is adjacent to a square with a pit
breeze((X,Y)) :-
    adjacent((X,Y), (X1,Y1)),
    pit((X1,Y1)).

% A square has glitter if gold is in the same square
glitter((X,Y)) :-
    gold((X,Y)).

% A square is safe if it does not have a pit or a wumpus
safe((X,Y)) :-
    \+ pit((X,Y)),
    \+ wumpus((X,Y)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAMPLE QUERIES (TRY THESE IN PROLOG)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Is there a stench at (2,1)?
% ?- stench((2,1)).

% Is there a breeze at (2,2)?
% ?- breeze((2,2)).

% Is there glitter at (3,3)?
% ?- glitter((3,3)).

% Find all squares with a breeze:
% ?- breeze(X).

% Find all squares with a stench:
% ?- stench(X).

% Find all safe squares:
% ?- safe(X).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHALLENGE QUERIES FOR STUDENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Where could the wumpus be inferred from stench?
% ?- stench((X,Y)).

% Which squares are dangerous?
% ?- pit(X).
% ?- wumpus(X).

% Explore relationships:
% ?- adjacent((2,2), X).