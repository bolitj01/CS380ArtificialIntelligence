%Comments are made with the % symbol
% Facts are made with predicates and arguments
parent(marie, travis).
% The . symbol is used to end a fact, rule, or query in Prolog. (like a ; in Java)
parent(marie, mandy).
parent(marie, michelle).
parent(marie, liz).
parent(marie, tommy).
parent(marie, robby).

% The spouses of marie's children
married(marie, robert).
married(mandy, jon).
married(travis, becky).
married(michelle, adam).
married(liz, jimmy).
married(tommy, haley).

% marie's mother
parent(anita, marie).

% Rules are made with predicates and variables
% :- is used to define a rule, which establishes a relationship between the variables in the rule.

% If X is married to Z and Z is a parent of Y, then X is also a parent of Y.
parent(P2, C) :- married(P1, P2), parent(P1, C).
% This is easier than stating the parent relationship for each of marie's children with marie's spouse, robert.

% If C is a child of P, then P is a parent of C.
child(C, P) :- parent(P, C).

% , is like a logical AND, meaning that both conditions must be true for the rule to be satisfied.
% ; is like a logical OR, meaning that at least one of the conditions must be true for the rule to be satisfied.
% \= means P can't be the same as C
sibling(C1, C2) :- 
    parent(P, C1), parent(P, C2), C1 \= C2; %share a parent
    married(C3, C2), sibling(C1, C3). % married to a sibling (in-law)
    % Why does 
    %sibling(C1, C3), married(C3, C2) % cause an infinite loop? 

grandparent(G, C) :- parent(G, P), parent(P, C).

%TODO Add rules for cousins and then add these parent-children facts:
% travis - noah
% travis - lydia
% travis - micah
% travis - luke
% mandy - zay
% mandy - kai

% Queries are made with predicates and variables
% The ?- symbol is used to ask a question about the facts and rules defined in the program.
% TODO Add these queries:
% Who are all the grandchildren?
% Who are all of robert's kids?
% Is tommy a sibling of becky?
% Is tommy a sibling of noah?