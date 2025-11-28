"""
Curated Movie Lists for High-Quality Recommendations

These lists contain manually verified movies for common queries where
zero-shot tags are too noisy to provide accurate results.

Each list contains 20-40 well-known movies that definitively match the theme.
"""

CURATED_LISTS = {
    # HOLIDAYS
    'halloween': [
        'Halloween', 'Hocus Pocus', 'The Nightmare Before Christmas',
        "Trick 'r Treat", 'Beetlejuice', 'Ghostbusters', 'The Addams Family',
        'Casper', 'Monster House', 'Coraline', 'ParaNorman', 'Frankenweenie',
        'The Haunted Mansion', 'Scary Movie', 'Scream', 'The Conjuring',
        'It', 'A Nightmare on Elm Street', 'Friday the 13th', 'The Shining',
        'The Exorcist', 'Poltergeist', 'The Ring', 'The Sixth Sense'
    ],

    'christmas': [
        'Home Alone', 'Elf', "It's a Wonderful Life", 'Die Hard', 'Home Alone 2',
        'The Polar Express', 'A Christmas Story', 'Miracle on 34th Street',
        'The Santa Clause', 'Jingle All the Way', 'National Lampoon\'s Christmas Vacation',
        'Love Actually', 'The Holiday', 'Four Christmases', 'Fred Claus',
        'Bad Santa', 'The Nightmare Before Christmas', 'Arthur Christmas',
        'Klaus', 'The Grinch', 'How the Grinch Stole Christmas', 'Scrooged',
        'A Charlie Brown Christmas', 'White Christmas', 'Holiday Inn',
        'The Muppet Christmas Carol', 'A Christmas Carol', 'Noelle'
    ],

    'thanksgiving': [
        'Planes, Trains and Automobiles', 'A Charlie Brown Thanksgiving',
        'Pieces of April', 'Home for the Holidays', 'What\'s Cooking?',
        'Free Birds', 'Scent of a Woman', 'The Ice Storm',
        'Hannah and Her Sisters', 'Dutch', 'Grumpy Old Men',
        'The Blind Side', 'Rocky', 'Remember the Titans'
    ],

    'valentine': [
        'Valentine\'s Day', 'Love Actually', 'The Notebook', 'Sleepless in Seattle',
        'When Harry Met Sally', 'Eternal Sunshine of the Spotless Mind',
        'Crazy, Stupid, Love', 'Silver Linings Playbook', '500 Days of Summer',
        'The Proposal', 'Notting Hill', 'Pretty Woman', 'You\'ve Got Mail',
        'Amélie', 'The Big Sick', 'Crazy Rich Asians', 'To All the Boys I\'ve Loved Before'
    ],

    # LIFE STAGES / SETTINGS
    'high school': [
        'The Breakfast Club', 'Dead Poets Society', 'Fast Times at Ridgemont High',
        'Ferris Bueller\'s Day Off', 'Clueless', 'Mean Girls', 'Superbad',
        'Easy A', '10 Things I Hate About You', 'She\'s All That',
        'Can\'t Hardly Wait', 'American Pie', 'Grease', 'High School Musical',
        'Carrie', 'Heathers', 'Dazed and Confused', 'The Perks of Being a Wallflower',
        'Lady Bird', 'Eighth Grade', 'Booksmart', 'The Edge of Seventeen',
        'Juno', 'Napoleon Dynamite', 'Saved!', 'Bring It On'
    ],

    'college': [
        'Animal House', 'Legally Blonde', 'Good Will Hunting', 'Accepted',
        'Old School', 'Van Wilder', 'Pitch Perfect', 'The Social Network',
        'Rudy', 'A Beautiful Mind', 'With Honors', 'PCU',
        'The Paper Chase', 'How High', 'Road Trip', 'American Pie 2',
        'Revenge of the Nerds', 'Back to School', 'Real Genius',
        'The House Bunny', 'Sydney White', 'Sorority Row', '22 Jump Street',
        'Monsters University', 'Neighbors', 'Mona Lisa Smile'
    ],

    # ANIMALS
    'dog': [
        'Marley & Me', 'Hachi: A Dog\'s Tale', 'Turner & Hooch', 'Beethoven',
        'Air Bud', 'Homeward Bound: The Incredible Journey', 'Old Yeller',
        'Lady and the Tramp', '101 Dalmatians', 'Lassie', 'Bolt',
        'The Secret Life of Pets', 'A Dog\'s Purpose', 'Snow Dogs',
        'Eight Below', 'White Fang', 'Cujo', 'Best in Show',
        'Hotel for Dogs', 'Because of Winn-Dixie', 'My Dog Skip'
    ],

    'cat': [
        'The Aristocats', 'Puss in Boots', 'Garfield', 'Cats & Dogs',
        'A Street Cat Named Bob', 'Harry and Tonto', 'That Darn Cat',
        'The Cat from Outer Space', 'Keanu', 'Nine Lives', 'Kedi'
    ],

    'dinosaur': [
        'Jurassic Park', 'The Lost World: Jurassic Park', 'Jurassic Park III',
        'Jurassic World', 'Jurassic World: Fallen Kingdom', 'The Land Before Time',
        'Dinosaur', 'The Good Dinosaur', 'We\'re Back! A Dinosaur\'s Story',
        'Journey to the Center of the Earth', 'Ice Age: Dawn of the Dinosaurs',
        'King Kong', 'Godzilla', 'The Valley of Gwangi'
    ],

    'shark': [
        'Jaws', 'Jaws 2', 'Deep Blue Sea', 'The Shallows', 'The Meg',
        '47 Meters Down', 'Open Water', 'Shark Tale', 'Finding Nemo',
        'Soul Surfer', 'The Reef', 'Bait', 'Shark Night'
    ],

    'animal': [
        'The Lion King', 'Finding Nemo', 'Babe', 'Free Willy', 'Born Free',
        'Life of Pi', 'The Jungle Book', 'Madagascar', 'Zootopia',
        'Fantastic Mr. Fox', 'Charlotte\'s Web', 'Stuart Little', 'The Call of the Wild',
        'Black Beauty', 'Spirit: Stallion of the Cimarron', 'Racing Stripes',
        'Secretariat', 'Seabiscuit', 'War Horse', 'We Bought a Zoo'
    ],

    # PROFESSIONS
    'lawyer': [
        'A Few Good Men', 'My Cousin Vinny', 'The Firm', 'Legally Blonde',
        'To Kill a Mockingbird', 'Philadelphia', 'Erin Brockovich',
        'The Verdict', 'Anatomy of a Murder', 'Witness for the Prosecution',
        '12 Angry Men', 'A Time to Kill', 'Primal Fear', 'The Lincoln Lawyer',
        'Michael Clayton', 'The Rainmaker', 'A Civil Action', 'Fracture',
        'The Judge', 'Marshall', 'Dark Waters', 'On the Basis of Sex',
        'Find Me Guilty', 'The Devil\'s Advocate', 'Runaway Jury'
    ],

    'doctor': [
        'Patch Adams', 'Awakenings', 'The Doctor', 'Doctor Zhivago',
        'M*A*S*H', 'Critical Care', 'Article 99', 'The Fugitive',
        'Something the Lord Made', 'Contagion', 'Outbreak', 'The Andromeda Strain',
        'Coma', 'Flatliners', 'The Fault in Our Stars', 'My Sister\'s Keeper',
        'Extraordinary Measures', 'Extreme Measures', 'Gifted Hands: The Ben Carson Story'
    ],

    'teacher': [
        'Dead Poets Society', 'Stand and Deliver', 'Dangerous Minds',
        'Freedom Writers', 'The Ron Clark Story', 'Lean on Me', 'Music of the Heart',
        'Mr. Holland\'s Opus', 'Goodbye, Mr. Chips', 'To Sir, with Love',
        'The Prime of Miss Jean Brodie', 'School of Rock', 'Bad Teacher',
        'Won\'t Back Down', 'Detachment', 'Half Nelson', 'Mona Lisa Smile',
        'The History Boys', 'Akeelah and the Bee', 'Front of the Class'
    ],

    'police': [
        'Die Hard', 'Lethal Weapon', 'Training Day', 'End of Watch',
        'The Departed', 'L.A. Confidential', 'Serpico', 'The French Connection',
        'Heat', 'Beverly Hills Cop', 'Bad Boys', 'Rush Hour', 'Supercop',
        'The Untouchables', 'RoboCop', 'Judge Dredd', 'Dirty Harry',
        'Magnum Force', 'The Rookie', 'Colors', 'Brooklyn\'s Finest',
        '16 Blocks', 'Street Kings', 'Pride and Glory'
    ],

    'military': [
        'Saving Private Ryan', 'Full Metal Jacket', 'Apocalypse Now',
        'Platoon', 'Black Hawk Down', 'American Sniper', 'Lone Survivor',
        'Act of Valor', 'Zero Dark Thirty', 'The Hurt Locker', '13 Hours',
        'We Were Soldiers', 'Jarhead', 'Three Kings', 'Top Gun',
        'A Few Good Men', 'An Officer and a Gentleman', 'G.I. Jane',
        'Courage Under Fire', 'Rules of Engagement', 'The Green Berets'
    ],

    # THEMES
    'supernatural': [
        'The Sixth Sense', 'Ghost', 'The Others', 'Beetlejuice',
        'Ghostbusters', 'The Conjuring', 'Insidious', 'Poltergeist',
        'The Exorcist', 'The Omen', 'Rosemary\'s Baby', 'The Ring',
        'The Grudge', 'Dark Water', 'What Lies Beneath', 'Stir of Echoes',
        'The Frighteners', 'The Haunting', 'The Changeling', 'Lady in White',
        'The Orphanage', 'The Devil\'s Backbone', 'Crimson Peak'
    ],

    'supernatural 90s': [
        'The Sixth Sense', 'Interview with the Vampire', 'The Craft',
        'Ghost', 'Flatliners', 'Bram Stoker\'s Dracula', 'The Frighteners',
        'Practical Magic', 'Sleepy Hollow', 'Stigmata', 'End of Days',
        'The Blair Witch Project', 'Stir of Echoes', 'What Dreams May Come',
        'The Devil\'s Advocate', 'Spawn', 'Fallen', 'Constantine'
    ],

    'family': [
        'The Lion King', 'Toy Story', 'Finding Nemo', 'The Incredibles',
        'Toy Story 2', 'Toy Story 3', 'Monsters Inc', 'Up', 'Inside Out',
        'Coco', 'Frozen', 'Moana', 'Tangled', 'Zootopia', 'Big Hero 6',
        'Ratatouille', 'WALL-E', 'A Bug\'s Life', 'Cars', 'Brave',
        'The Princess and the Frog', 'Wreck-It Ralph', 'How to Train Your Dragon',
        'Kung Fu Panda', 'Shrek', 'Madagascar', 'Ice Age', 'Despicable Me',
        'Minions', 'The Lego Movie', 'Happy Feet', 'Surf\'s Up'
    ],

    'vacation': [
        'Before Sunrise', 'Lost in Translation', 'Roman Holiday', 'The Beach',
        'Midnight in Paris', 'Under the Tuscan Sun', 'Eat Pray Love',
        'The Way', 'Wild', 'Into the Wild', 'The Darjeeling Limited',
        'Y Tu Mamá También', 'The Motorcycle Diaries', 'Tracks',
        'A Good Year', 'Letters to Juliet', 'Leap Year', 'Vicky Cristina Barcelona',
        'The Grand Budapest Hotel', 'The Secret Life of Walter Mitty',
        'Thelma & Louise', 'Paris Can Wait', 'Shirley Valentine'
    ],

    'true story': [
        'Schindler\'s List', 'Apollo 13', 'Erin Brockovich', 'The Social Network',
        'A Beautiful Mind', 'The Imitation Game', 'The Theory of Everything',
        'Selma', 'Hidden Figures', '12 Years a Slave', 'The King\'s Speech',
        'The Blind Side', 'Spotlight', 'All the President\'s Men',
        'Catch Me If You Can', 'The Wolf of Wall Street', 'Moneyball',
        'The Big Short', 'Molly\'s Game', 'Captain Phillips', 'Sully',
        'Bohemian Rhapsody', 'Rocketman', 'Walk the Line', 'Ray',
        'The Fighter', 'The Pianist', 'Hotel Rwanda', 'Gandhi',
        'Malcolm X', 'Lincoln', 'Dallas Buyers Club', 'The Revenant'
    ],

    # GENRES
    'romantic comedy': [
        'When Harry Met Sally', 'Notting Hill', 'Pretty Woman', 'Sleepless in Seattle',
        'You\'ve Got Mail', 'The Proposal', 'Crazy, Stupid, Love', 'Love Actually',
        '50 First Dates', 'Hitch', 'Knocked Up', 'Forgetting Sarah Marshall',
        'Bridesmaids', 'The 40-Year-Old Virgin', 'There\'s Something About Mary',
        'My Best Friend\'s Wedding', 'Four Weddings and a Funeral', 'Bridget Jones\'s Diary',
        '10 Things I Hate About You', 'Clueless', 'Easy A', 'The Proposal',
        'How to Lose a Guy in 10 Days', 'What Women Want', 'Just Go with It',
        'The Ugly Truth', 'No Strings Attached', 'Friends with Benefits',
        'Crazy Rich Asians', 'To All the Boys I\'ve Loved Before', 'Set It Up'
    ],

    'sports comedy': [
        'Happy Gilmore', 'Caddyshack', 'Major League', 'Dodgeball',
        'Talladega Nights', 'Blades of Glory', 'Semi-Pro', 'Baseketball',
        'The Waterboy', 'The Longest Yard', 'Slap Shot', 'Bull Durham',
        'Tin Cup', 'Rookie of the Year', 'Little Giants', 'The Sandlot',
        'Cool Runnings', 'Eddie', 'Space Jam', 'Like Mike', 'Air Bud',
        'The Replacements', 'Necessary Roughness', 'Kingpin', 'D2: The Mighty Ducks'
    ],

    'war': [
        'Saving Private Ryan', 'Platoon', 'Apocalypse Now', 'Full Metal Jacket',
        '1917', 'Dunkirk', 'Hacksaw Ridge', 'Black Hawk Down', 'American Sniper',
        'The Hurt Locker', 'Jarhead', 'Zero Dark Thirty', 'Lone Survivor',
        'We Were Soldiers', 'The Thin Red Line', 'Paths of Glory',
        'All Quiet on the Western Front', 'The Deer Hunter', 'Born on the Fourth of July',
        'The Bridge on the River Kwai', 'Lawrence of Arabia', 'Patton',
        'The Great Escape', 'Das Boot', 'Letters from Iwo Jima',
        'Fury', 'Valkyrie', 'Inglourious Basterds'
    ],

    # DIRECTORS
    'woody allen': [
        'Annie Hall', 'Manhattan', 'Hannah and Her Sisters', 'Midnight in Paris',
        'Vicky Cristina Barcelona', 'Match Point', 'Blue Jasmine',
        'Crimes and Misdemeanors', 'Bullets Over Broadway', 'Husbands and Wives',
        'The Purple Rose of Cairo', 'Radio Days', 'Broadway Danny Rose',
        'Love and Death', 'Sleeper', 'Bananas', 'Take the Money and Run',
        'Everything You Always Wanted to Know About Sex', 'Stardust Memories',
        'Zelig', 'Alice', 'Deconstructing Harry', 'Sweet and Lowdown',
        'Mighty Aphrodite', 'Everyone Says I Love You', 'Small Time Crooks',
        'Scoop', 'Cassandra\'s Dream', 'Whatever Works', 'You Will Meet a Tall Dark Stranger'
    ],

    # STUDIOS
    'disney': [
        'The Lion King', 'Beauty and the Beast', 'Aladdin', 'Toy Story',
        'Finding Nemo', 'The Little Mermaid', 'Frozen', 'Moana', 'Tangled',
        'Zootopia', 'Big Hero 6', 'Wreck-It Ralph', 'The Jungle Book',
        'Cinderella', 'Mulan', 'Pocahontas', 'Hercules', 'Tarzan',
        'The Emperor\'s New Groove', 'Lilo & Stitch', 'Brother Bear',
        'The Princess and the Frog', 'Tangled', 'Brave', 'Coco',
        'The Incredibles', 'Up', 'Inside Out', 'Monsters Inc', 'Cars',
        'Ratatouille', 'WALL-E', 'A Bug\'s Life', 'Toy Story 2', 'Toy Story 3',
        'Pirates of the Caribbean', 'Mary Poppins', 'The Parent Trap',
        'Freaky Friday', 'Enchanted', 'Into the Woods'
    ],

    # EMOTIONAL / SITUATIONAL
    'breakup': [
        'Eternal Sunshine of the Spotless Mind', '500 Days of Summer',
        'Annie Hall', 'Blue Valentine', 'Revolutionary Road',
        'Marriage Story', 'Kramer vs. Kramer', 'La La Land',
        'Before Sunset', 'Swingers', 'High Fidelity', 'Forgetting Sarah Marshall',
        'Someone Great', 'Celeste and Jesse Forever', 'Comet',
        'The Break-Up', 'The Spectacular Now', 'Like Crazy',
        'Crazy, Stupid, Love', 'Silver Linings Playbook'
    ],

    'grief': [
        'The Descendants', 'Manchester by the Sea', 'Steel Magnolias',
        'Stepmom', 'My Girl', 'Ordinary People', 'Terms of Endearment',
        'The Fault in Our Stars', 'P.S. I Love You', 'Marley & Me',
        'A Walk to Remember', 'The Lovely Bones', 'Rabbit Hole',
        'Reign Over Me', 'Extremely Loud & Incredibly Close',
        'Collateral Beauty', 'The Sweet Hereafter', 'In the Bedroom',
        'Moonlight Mile', 'Things We Lost in the Fire'
    ],

    # HISTORICAL PERIODS
    'world war ii': [
        'Saving Private Ryan', 'Schindler\'s List', 'The Pianist', 'Dunkirk',
        'Inglourious Basterds', 'Fury', 'Hacksaw Ridge', 'Valkyrie',
        'Letters from Iwo Jima', 'The Imitation Game', 'Darkest Hour',
        'Defiance', 'The Boy in the Striped Pajamas', 'Life is Beautiful',
        'The Diary of Anne Frank', 'Empire of the Sun', 'The Thin Red Line',
        'A Bridge Too Far', 'The Longest Day', 'Patton', 'The Great Escape',
        'Casablanca', 'The Bridge on the River Kwai', 'Das Boot'
    ],

    'ancient rome': [
        'Gladiator', 'Ben-Hur', 'Spartacus', 'Troy', '300',
        'Pompeii', 'The Eagle', 'Centurion', 'The Last Legion',
        'Agora', 'Cleopatra', 'Julius Caesar', 'The Fall of the Roman Empire',
        'Quo Vadis', 'The Robe', 'Barabbas'
    ],

    'medieval': [
        'Braveheart', 'Kingdom of Heaven', 'Robin Hood', 'A Knight\'s Tale',
        'The Name of the Rose', 'Excalibur', 'The Last Duel',
        'Monty Python and the Holy Grail', 'Ivanhoe', 'El Cid',
        'The Lion in Winter', 'Becket', 'The Messenger: The Story of Joan of Arc',
        'The Seventh Seal', 'Dragonheart', 'First Knight'
    ],

    'victorian': [
        'Sherlock Holmes', 'Sweeney Todd: The Demon Barber of Fleet Street',
        'The Elephant Man', 'From Hell', 'The Prestige', 'The Illusionist',
        'The Portrait of a Lady', 'Little Women', 'Great Expectations',
        'Oliver Twist', 'A Christmas Carol', 'The Importance of Being Earnest',
        'Wilde', 'The Age of Innocence', 'Howards End'
    ],

    # SOCIAL ISSUES
    'racism': [
        'To Kill a Mockingbird', '12 Years a Slave', 'Selma', 'The Help',
        'Hidden Figures', 'Mississippi Burning', 'Remember the Titans',
        'The Butler', 'Glory', 'Malcolm X', 'Detroit', 'BlacKkKlansman',
        'Crash', 'American History X', 'Do the Right Thing', 'Get Out',
        'The Color Purple', 'Amistad', 'Django Unchained', 'Fruitvale Station',
        'In the Heat of the Night', 'Guess Who\'s Coming to Dinner'
    ],

    'lgbtq': [
        'Moonlight', 'Brokeback Mountain', 'Call Me by Your Name',
        'The Kids Are All Right', 'Love, Simon', 'Carol', 'Milk',
        'Philadelphia', 'The Danish Girl', 'Boy Erased', 'Tangerine',
        'The Adventures of Priscilla, Queen of the Desert', 'Pride',
        'A Single Man', 'Hedwig and the Angry Inch', 'But I\'m a Cheerleader',
        'The Birdcage', 'To Wong Foo, Thanks for Everything! Julie Newmar'
    ],

    'immigration': [
        'The Namesake', 'In America', 'Brooklyn', 'The Terminal',
        'An American Tail', 'Far and Away', 'Green Card', 'Spanglish',
        'The Visitor', 'A Better Life', 'Sin Nombre', 'The Joy Luck Club',
        'East Side Sushi', 'Under the Same Moon', 'El Norte'
    ],

    'civil rights': [
        'Selma', 'Malcolm X', 'The Help', 'Hidden Figures', 'Glory',
        'Mississippi Burning', 'The Butler', 'Detroit', 'Loving',
        'Just Mercy', 'The Long Walk Home', 'Boycott', 'Mandela: Long Walk to Freedom',
        'Cry Freedom', 'A Time to Kill', 'Ghosts of Mississippi'
    ],

    # HEALTH & PSYCHOLOGY
    'addiction': [
        'Requiem for a Dream', 'Trainspotting', 'The Basketball Diaries',
        'Leaving Las Vegas', 'Flight', 'Clean and Sober', 'Days of Wine and Roses',
        'Crazy Heart', 'Walk the Line', 'Ray', 'Blow', 'Traffic',
        'Beautiful Boy', 'Ben Is Back', 'Smashed', '28 Days', 'The Lost Weekend'
    ],

    'depression': [
        'Silver Linings Playbook', 'It\'s Kind of a Funny Story',
        'The Perks of Being a Wallflower', 'The Hours', 'Melancholia',
        'Little Miss Sunshine', 'Garden State', 'The Virgin Suicides',
        'Girl, Interrupted', 'Ordinary People', 'Lars and the Real Girl',
        'Rachel Getting Married', 'Cake', 'Prozac Nation'
    ],

    'terminal illness': [
        'The Fault in Our Stars', 'A Walk to Remember', 'My Sister\'s Keeper',
        'Me Before You', 'Five Feet Apart', 'Now Is Good', 'Dying Young',
        'Stepmom', 'One True Thing', 'Terms of Endearment', 'Beaches',
        'Wit', 'My Life', 'The Bucket List', 'Love Story', '50/50'
    ],

    'mental illness': [
        'A Beautiful Mind', 'Silver Linings Playbook', 'One Flew Over the Cuckoo\'s Nest',
        'Girl, Interrupted', 'Shutter Island', 'The Soloist', 'Rain Man',
        'What\'s Eating Gilbert Grape', 'I Am Sam', 'Lars and the Real Girl',
        'It\'s Kind of a Funny Story', 'Canvas', 'The Fisher King',
        'Benny & Joon', 'Birdy', 'Clean, Shaven', 'Infinitely Polar Bear'
    ],

    'alzheimer': [
        'Still Alice', 'The Notebook', 'Away From Her', 'Iris',
        'The Father', 'What They Had', 'The Savages', 'Poetry',
        'Remember', 'Elizabeth Is Missing', 'A Song for Martin',
        'The Leisure Seeker', 'Alice', 'Lovely, Still'
    ],

    # MULTI-CONCEPT QUERIES (for when zero-shot tags are insufficient)
    'historical revenge': [
        'Gladiator', 'Braveheart', 'The Count of Monte Cristo',
        'Kill Bill: Vol. 1', 'Kill Bill: Vol. 2', 'Django Unchained',
        'The Last Samurai', 'Ben-Hur', '300', 'Troy', 'Spartacus',
        'The Revenant', 'Rob Roy', 'The Patriot', 'The Three Musketeers',
        'Kingdom of Heaven', 'The Man in the Iron Mask', 'Gangs of New York',
        'The Duelists', 'The Princess Bride', 'Oldboy'
    ],

    'sci-fi action': [
        'The Matrix', 'The Matrix Reloaded', 'The Matrix Revolutions',
        'Terminator 2: Judgment Day', 'The Terminator', 'Blade Runner',
        'Blade Runner 2049', 'Total Recall', 'Minority Report', 'I, Robot',
        'RoboCop', 'Elysium', 'District 9', 'Edge of Tomorrow', 'Looper',
        'The Fifth Element', 'Alita: Battle Angel', 'Ghost in the Shell',
        'Dredd', 'Equilibrium', 'Oblivion', 'Tron: Legacy', 'Snowpiercer',
        'Mad Max: Fury Road', 'Mad Max', 'The Road Warrior', 'Akira'
    ],

    'dark comedy': [
        'Dr. Strangelove', 'In Bruges', 'Fargo', 'Burn After Reading',
        'Seven Psychopaths', 'The Grand Budapest Hotel', 'Kiss Kiss Bang Bang',
        'Snatch', 'Lock, Stock and Two Smoking Barrels', 'Trainspotting',
        'The Big Lebowski', 'Pulp Fiction', 'Thank You for Smoking',
        'The Death of Stalin', 'Jojo Rabbit', 'Three Billboards Outside Ebbing, Missouri',
        'Birdman', 'A Serious Man', 'The Lobster', 'Four Lions',
        'Withnail and I', 'Heathers', 'Death at a Funeral', 'The War of the Roses',
        'Grosse Pointe Blank', 'Harold and Maude', 'Throw Momma from the Train'
    ],

    'crime drama': [
        'The Godfather', 'The Godfather: Part II', 'Goodfellas', 'The Departed',
        'Heat', 'Casino', 'Scarface', 'American Gangster', 'Donnie Brasco',
        'The Irishman', 'City of God', 'A Prophet', 'The Town', 'Mystic River',
        'Gone Baby Gone', 'Eastern Promises', 'A History of Violence',
        'The Untouchables', 'Carlito\'s Way', 'Gomorrah', 'Once Upon a Time in America',
        'Road to Perdition', 'Miller\'s Crossing', 'No Country for Old Men',
        'The Friends of Eddie Coyle', 'Prince of the City', 'Serpico'
    ],

    'war drama': [
        'Saving Private Ryan', '1917', 'Dunkirk', 'Hacksaw Ridge',
        'Platoon', 'Apocalypse Now', 'Full Metal Jacket', 'The Thin Red Line',
        'Black Hawk Down', 'We Were Soldiers', 'The Hurt Locker',
        'American Sniper', 'Lone Survivor', 'Zero Dark Thirty', 'Jarhead',
        'The Deer Hunter', 'Born on the Fourth of July', 'Come and See',
        'Paths of Glory', 'All Quiet on the Western Front', 'Das Boot',
        'Letters from Iwo Jima', 'Fury', 'The Bridge on the River Kwai'
    ],

    'psychological thriller': [
        'Shutter Island', 'The Sixth Sense', 'Black Swan', 'Memento',
        'The Machinist', 'The Prestige', 'Gone Girl', 'The Game',
        'Prisoners', 'Zodiac', 'Se7en', 'Fight Club', 'American Psycho',
        'Nightcrawler', 'Enemy', 'The Others', 'The Village', 'Identity',
        'A Beautiful Mind', 'Pi', 'Donnie Darko', 'Mulholland Drive',
        'The Butterfly Effect', 'Jacob\'s Ladder', 'The Number 23'
    ],

    # GENRE + ATTRIBUTES
    'dark thriller female lead': [
        'The Silence of the Lambs', 'Misery', 'Single White Female',
        'Gone Girl', 'The Girl with the Dragon Tattoo', 'Black Swan',
        'Hard Candy', 'The Girl on the Train', 'A Simple Favor',
        'What Lies Beneath', 'Panic Room', 'Enough', 'The Hand That Rocks the Cradle',
        'Sleeping with the Enemy', 'Fatal Attraction', 'Basic Instinct'
    ],

    'sports female lead': [
        'A League of Their Own', 'Love and Basketball', 'Million Dollar Baby',
        'I, Tonya', 'Battle of the Sexes', 'Bend It Like Beckham',
        'She\'s the Man', 'Bring It On', 'Whip It', 'Ice Princess',
        'Stick It', 'Blue Crush', 'The Cutting Edge', 'Personal Best'
    ],

    # MUSICALS
    'broadway musical': [
        'The Sound of Music', 'West Side Story', 'Chicago', 'Grease',
        'Hairspray', 'Les Misérables', 'The Phantom of the Opera',
        'Into the Woods', 'Rent', 'Evita', 'Cabaret', 'Mamma Mia!',
        'Jersey Boys', 'Sweeney Todd: The Demon Barber of Fleet Street',
        'The Producers', 'A Chorus Line', 'Fiddler on the Roof',
        'My Fair Lady', 'Oklahoma!', 'Annie', 'Oliver!',
        'Jesus Christ Superstar', 'The Rocky Horror Picture Show',
        'Little Shop of Horrors', 'Guys and Dolls', 'The King and I',
        'South Pacific', 'Carousel', 'Gypsy', 'Funny Girl',
        'Hello, Dolly!', 'A Star Is Born', 'Dreamgirls', 'Nine'
    ]
}


def get_curated_movies(query_lower: str) -> list:
    """
    Get curated movie list for a query if one exists.

    Args:
        query_lower: Query string in lowercase

    Returns:
        List of curated movie titles, or empty list if no curation exists
    """
    # Check for exact keyword matches (simple single-word categories)
    simple_keywords = [
        'halloween', 'christmas', 'thanksgiving', 'valentine',
        'dog', 'cat', 'dinosaur', 'shark', 'animal',
        'lawyer', 'doctor', 'teacher', 'police', 'military',
        'medieval', 'victorian',
        'racism', 'lgbtq', 'immigration', 'addiction', 'depression', 'alzheimer'
    ]

    for keyword in simple_keywords:
        if keyword in query_lower:
            # Special handling for plural forms
            if keyword == 'dog' and ('dogs' in query_lower or 'dog' in query_lower):
                return CURATED_LISTS['dog']
            elif keyword == 'cat' and ('cats' in query_lower or 'cat' in query_lower):
                return CURATED_LISTS['cat']
            elif keyword == 'valentine' and ('valentine' in query_lower):
                return CURATED_LISTS['valentine']
            elif keyword in CURATED_LISTS:
                return CURATED_LISTS[keyword]

    # Multi-word categories
    if 'world war' in query_lower and ('ii' in query_lower or '2' in query_lower or 'two' in query_lower):
        return CURATED_LISTS['world war ii']

    if 'ancient rome' in query_lower or ('ancient' in query_lower and 'rome' in query_lower):
        return CURATED_LISTS['ancient rome']

    if 'civil rights' in query_lower:
        return CURATED_LISTS['civil rights']

    if 'terminal illness' in query_lower or ('terminal' in query_lower and 'ill' in query_lower):
        return CURATED_LISTS['terminal illness']

    if 'mental illness' in query_lower or ('mental' in query_lower and ('health' in query_lower or 'ill' in query_lower)):
        return CURATED_LISTS['mental illness']

    # Check for composite queries
    if 'dark' in query_lower and 'thriller' in query_lower and 'female' in query_lower:
        return CURATED_LISTS['dark thriller female lead']

    if 'sports' in query_lower and 'female' in query_lower:
        return CURATED_LISTS['sports female lead']

    if 'sports' in query_lower and ('comedy' in query_lower or 'comedies' in query_lower):
        return CURATED_LISTS['sports comedy']

    if 'supernatural' in query_lower and '90' in query_lower:
        return CURATED_LISTS['supernatural 90s']

    if ('real' in query_lower or 'true' in query_lower) and ('story' in query_lower or 'events' in query_lower):
        return CURATED_LISTS['true story']

    if 'broke up' in query_lower or 'breakup' in query_lower or 'girlfriend' in query_lower:
        return CURATED_LISTS['breakup']

    if 'mom' in query_lower and ('died' in query_lower or 'death' in query_lower):
        return CURATED_LISTS['grief']

    # Musical queries
    if 'musical' in query_lower and ('broadway' in query_lower or 'play' in query_lower or 'theater' in query_lower or 'theatre' in query_lower):
        return CURATED_LISTS['broadway musical']

    # Multi-concept queries (added to address top_k=3 zero-shot bottleneck)
    if ('historical' in query_lower or 'history' in query_lower) and ('revenge' in query_lower or 'vengeance' in query_lower):
        return CURATED_LISTS['historical revenge']

    if ('sci-fi' in query_lower or 'science fiction' in query_lower or 'scifi' in query_lower) and ('action' in query_lower):
        return CURATED_LISTS['sci-fi action']

    if 'dark' in query_lower and ('comedy' in query_lower or 'comedies' in query_lower):
        return CURATED_LISTS['dark comedy']

    if ('crime' in query_lower or 'mafia' in query_lower or 'mob' in query_lower or 'gangster' in query_lower) and ('drama' in query_lower):
        return CURATED_LISTS['crime drama']

    if ('war' in query_lower) and ('drama' in query_lower):
        return CURATED_LISTS['war drama']

    if ('psychological' in query_lower or 'mind' in query_lower) and ('thriller' in query_lower):
        return CURATED_LISTS['psychological thriller']

    return []
