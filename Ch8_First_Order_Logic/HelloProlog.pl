%Prolog knowledge base about me
name(tommy).
age(30).
years_of_teaching(tommy, 6).
likes(tommy, jesus).
likes(tommy, teaching).
likes(tommy, technology).
likes(tommy, volleyball).

% Anything I like is cool
cool(X) :- likes(tommy, X).
experienced(Person) :-
    years_of_teaching(Person, Y),
    Y > 5.