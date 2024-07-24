import React, { createContext, useState, useEffect, useContext } from 'react';
import { getAuth, signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { doc, getFirestore, setDoc,getDoc } from 'firebase/firestore';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const auth = getAuth()
  const db = getFirestore()

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        // User is signed in
        setUser(user);

        // check if in firebase the record exists , else create it 
        const userDoc = doc(db, 'users', user.uid);
        const userDocSnapshot = await getDoc(userDoc);
        if (!userDocSnapshot.exists()) {
        
          await setDoc(userDoc, {
              user_id: user.uid,
              nationality:"",
              age:0,
              gender:"",
              experience:{
                decision_maker:false,
                volunteer:false,
                victim:false
              },
              consent_agree: true,
              training_available: 1,
              current_tree_level: 0,
              gameEnded:false,
              responses: {
                tree_0: {
                  points: [],
                  tree_score:0,
                  tree_done:false,
                  isCorrectlyAnswered:0,
                  isWronglyAnswered:0,
                  isGatherAdditionalDataRequested:0,
                  isCompleted: [false,false,false,false,false],
                  gotWronglyAnswered:[false,false,false,false,false],
                  gotGatherAdditionalDataRequested: [false,false,false,false,false],
                  tree: {
                    info: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    human: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    damage: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    satellite: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    'drone-damage': {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    }
                  }
                },
                tree_1: {
                  points: [],
                  tree_score:0,
                  tree_done:false,
                  isCorrectlyAnswered:0,
                  isWronglyAnswered:0,
                  isGatherAdditionalDataRequested:0,
                  isCompleted: [false,false,false,false,false],
                  gotWronglyAnswered:[false,false,false,false,false],
                  gotGatherAdditionalDataRequested: [false,false,false,false,false],
                  tree: {
                    info: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    human: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    damage: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    satellite: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    'drone-damage': {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    }
                  }
                },
                tree_2: {
                  points: [],
                  tree_score:0,
                  tree_done:false,
                  isCorrectlyAnswered:0,
                  isWronglyAnswered:0,
                  isGatherAdditionalDataRequested:0,
                  isCompleted: [false,false,false,false,false],
                  gotWronglyAnswered:[false,false,false,false,false],
                  gotGatherAdditionalDataRequested: [false,false,false,false,false],
                  tree: {
                    info: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    human: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    damage: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    satellite: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    'drone-damage': {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    }
                  }
                },
                tree_3: {
                  points: [],
                  tree_score:0,
                  tree_done:false,
                  isCorrectlyAnswered:0,
                  isWronglyAnswered:0,
                  isGatherAdditionalDataRequested:0,
                  isCompleted: [false,false,false,false,false],
                  gotWronglyAnswered:[false,false,false,false,false],
                  gotGatherAdditionalDataRequested: [false,false,false,false,false],
                  tree: {
                    info: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    human: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    damage: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    satellite: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    'drone-damage': {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    }
                  }
                },
                tree_4: {
                  points: [],
                  tree_score:0,
                  tree_done:false,
                  isCorrectlyAnswered:0,
                  isWronglyAnswered:0,
                  isGatherAdditionalDataRequested:0,
                  isCompleted: [false,false,false,false,false],
                  gotWronglyAnswered:[false,false,false,false,false],
                  gotGatherAdditionalDataRequested: [false,false,false,false,false],
                  tree: {
                    info: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    human: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    damage: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    satellite: {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    },
                    'drone-damage': {
                      user_answer: [],
                      availableAdditionalData: 5,
                      question_id: [],
                    }
                  }
                }
              },
              score: 0,
              CorrectlyAnswered:0,
              WronglyAnswered:0,
              GatherAdditionalDataRequested:0,
              number_of_completed_trees: 0,
              number_of_failed_trees: 0,
              feedback: ""
          }, { merge: true });
          console.log('User record created ')
        }else{
          console.log('User Data : ',userDocSnapshot.data())
        }
      } else {
        // Sign in anonymously
        const userCredential = await signInAnonymously(auth);
        const newUser = userCredential.user;
        setUser(newUser);

        const userDoc = doc(db, 'users', newUser.uid);
        
        await setDoc(userDoc, {
            user_id: user.uid,
            nationality:"",
            age:0,
            gender:"",
            experience:{
              decision_maker:false,
              volunteer:false,
              victim:false
            },
            consent_agree: true,
            training_available: 1,
            gameEnded:false,
            current_tree_level: 0,
            responses: {
              tree_0: {
                points: [],
                tree_done:false,
                tree_score:0,
                isCorrectlyAnswered:0,
                isWronglyAnswered:0,
                isGatherAdditionalDataRequested:0,
                isCompleted: [false,false,false,false,false],
                gotWronglyAnswered:[false,false,false,false,false],
                gotGatherAdditionalDataRequested: [false,false,false,false,false],
                tree: {
                  info: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  human: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  satellite: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_no_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                }
              },
              tree_1: {
                points: [],
                tree_done:false,
                tree_score:0,
                isCorrectlyAnswered:0,
                isWronglyAnswered:0,
                isGatherAdditionalDataRequested:0,
                isCompleted: [false,false,false,false,false],
                gotWronglyAnswered:[false,false,false,false,false],
                gotGatherAdditionalDataRequested: [false,false,false,false,false],
                tree: {
                  info: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  human: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  satellite: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_no_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                }
              },
              tree_2: {
                points: [],
                tree_done:false,
                tree_score:0,
                isCorrectlyAnswered:0,
                isWronglyAnswered:0,
                isGatherAdditionalDataRequested:0,
                isCompleted: [false,false,false,false,false],
                gotWronglyAnswered:[false,false,false,false,false],
                gotGatherAdditionalDataRequested: [false,false,false,false,false],
                tree: {
                  info: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  human: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  satellite: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_no_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                }
              },
              tree_3: {
                points: [],
                tree_done:false,
                tree_score:0,
                isCorrectlyAnswered:0,
                isWronglyAnswered:0,
                isGatherAdditionalDataRequested:0,
                isCompleted: [false,false,false,false,false],
                gotWronglyAnswered:[false,false,false,false,false],
                gotGatherAdditionalDataRequested: [false,false,false,false,false],
                tree: {
                  info: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  human: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  satellite: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_no_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                }
              },
              tree_4: {
                points: [],
                tree_done:false,
                tree_score:0,
                isCorrectlyAnswered:0,
                isWronglyAnswered:0,
                isGatherAdditionalDataRequested:0,
                isCompleted: [false,false,false,false,false],
                gotWronglyAnswered:[false,false,false,false,false],
                gotGatherAdditionalDataRequested: [false,false,false,false,false],
                tree: {
                  info: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  human: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  satellite: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                  drone_no_damage: {
                    user_answer: [],
                    availableAdditionalData: 5,
                    question_id: [],
                  },
                }
              }
            },
            score: 0,
            CorrectlyAnswered:0,
            WronglyAnswered:0,
            GatherAdditionalDataRequested:0,
            number_of_completed_trees: 0,
            number_of_failed_trees: 0,
            feedback: ""
        }, { merge: true });
         console.log('User record created ')
      }
      setLoading(false);
    });

    return unsubscribe;
  }, [auth, db]);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <AuthContext.Provider value={{ user }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
