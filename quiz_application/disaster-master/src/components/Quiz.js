import React, { useState, useEffect } from 'react';
import useFirestore from '../hooks/useFirestore';
import Question from './Question';
import Loader from './Loader';
// import Score from './Score';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { doc, getDoc, updateDoc } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';
import { checkNavigationHelper } from '../services/navigationHelper';


const MAPPING_DICT = {
  "info": {
    0: "Actionable Data",
    1: "No Response Necessary",
    100: "c"
  },
  "human": {
    0: "Actionable Data",
    1: "Actionable Data",
    2: "Actionable Data",
    3: "Actionable Data",
    4: "No Response Necessary",
    100: "Gather Additional Data"
  },
  "damage": {
    0: "Actionable Data",
    1: "Actionable Data",
    2: "No Response Necessary",
    100: "Gather Additional Data"
  },
  "satellite": {
    0: "No Response Necessary",
    1: "Actionable Data",
    100: "Gather Additional Data"
  },
  "drone-damage": {
    0: "No Response Necessary",
    1: "Actionable Data",
    100: "Gather Additional Data"
  }
}

const Quiz = ({ task, phase }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [userData, setUserData] = useState(null);
  const [currentTreeLevel, setCurrentTreeLevel] = useState(null);
  const [availableTraining, setAvailableTraining] = useState(null);
  const [availableValidation, setAvailableValidation] = useState(null);
  const tasksList = ["info", "human", "damage", "satellite", "drone-damage"]
  const phaseList = ["train", "val"]
  const { storedQuestion, loading } = useFirestore(task, phase);
  const [score, setScore] = useState(0);

  const fetchUserData = async () => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();
    setAvailableTraining(userDocData.training_available)
    setAvailableValidation(userDocData.validation_pending)
    setUserData(userDocData);
    setCurrentTreeLevel(userDocData.current_tree_level);
  };

  useEffect(() => {
    fetchUserData();
  }, [user.uid]);

  const checkIfContinueTraining = async () => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();
    const treeLevel = userDocData.training_available - 1;
    if (userDocData.training_available !== 0) {
      if (task === "drone-damage") {
        await updateDoc(userDoc, {
          training_available: treeLevel
        });
        fetchUserData()
        if (treeLevel === 0){
          return false
        }
        else{
          return true // means continue showing training data 
        }
      }
      else {
        fetchUserData()
        return true // continue showing training data
      }
    }else{
      fetchUserData()
      return false // stop showing training data
    }
  };

  const handleValidationPhase = async (question, userAnswer, isCorrect) => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();
  
    if (userDocData.validation_pending === 0) {
      navigate("/results");
      return;
    }
  
    let currentTree = userDocData.responses[`tree_${userDocData.current_tree_level}`];
  
    if (isCorrect) {
      handleCorrectAnswer(userDoc, userDocData, question, userAnswer, currentTree);
    } else if (userAnswer === "Gather Additional Data") {
      handleGatherAdditionalData(userDoc, userDocData, question, userAnswer, currentTree);
    } else if (userAnswer === "No Response Necessary") {
      handleNoResponseNecessary(userDoc, userDocData, question, userAnswer, currentTree);
    }
    
  };
  
  
  const handleCorrectAnswer = async (userDoc, userDocData, question, userAnswer, currentTree) => {
    if (task === "drone-damage") {
      await updateUserDoc(userDoc, {
        validation_pending: userDocData.validation_pending - 1,
        current_tree_level: userDocData.current_tree_level + 1,
        number_of_completed_trees: userDocData.number_of_completed_trees + 1,
      });
    }
  
    currentTree.points.push(1);
    currentTree.tree[task].question_id.push(question.question_id);
    currentTree.tree[task].user_answer.push(userAnswer);
  
    const updates = {
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.question_id`]: currentTree.tree[task].question_id,
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.user_answer`]: currentTree.tree[task].user_answer,
      [`responses.tree_${userDocData.current_tree_level}.points`]: currentTree.points,
    };
  
    if (task === "drone-damage") {
      updates[`responses.tree_${userDocData.current_tree_level}.isCompleted`] = true;
    }
  
    await updateUserDoc(userDoc, updates);

    const { path, state }  = checkNavigationHelper(task, phase);
    navigate(userDocData.validation_pending === 0 ? "/results" : path, { state });
  };
  
  const handleGatherAdditionalData = async (userDoc, userDocData, question, userAnswer, currentTree) => {
    if (currentTree.tree[task].availableAdditionalData === 0) {
      alert("You have used up all Additional Data Credits for this task, Please Take a Conclusive Decision on the Task!");
      return;
    }
  
    currentTree.points.push(-1);
    currentTree.tree[task].availableAdditionalData -= 1;
  
    await updateUserDoc(userDoc, {
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.question_id`]: currentTree.tree[task].question_id,
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.user_answer`]: currentTree.tree[task].user_answer,
      [`responses.tree_${userDocData.current_tree_level}.points`]: currentTree.points,
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.availableAdditionalData`]: currentTree.tree[task].availableAdditionalData,
    });
  
    let previousTaskIndex = tasksList.indexOf(task) - 1;
    if (previousTaskIndex <0){
      previousTaskIndex = 0
    }
    let newTask = tasksList[previousTaskIndex]
    const { path, state } = checkNavigationHelper(newTask, phase);
    navigate('/loading', { state: { task: newTask, phase: phase } });
  };
  
  const handleNoResponseNecessary = async (userDoc, userDocData, question, userAnswer, currentTree) => {
    if (task === "drone-damage") {
      await updateUserDoc(userDoc, {
        validation_pending: userDocData.validation_pending - 1,
        current_tree_level: userDocData.current_tree_level + 1,
        number_of_failed_trees: userDocData.number_of_failed_trees + 1,
      });
    }
  
    currentTree.points.push(0);
    currentTree.tree[task].question_id.push(question.question_id);
    currentTree.tree[task].user_answer.push(userAnswer);
  
    await updateUserDoc(userDoc, {
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.question_id`]: currentTree.tree[task].question_id,
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.user_answer`]: currentTree.tree[task].user_answer,
      [`responses.tree_${userDocData.current_tree_level}.points`]: currentTree.points,
      [`responses.tree_${userDocData.current_tree_level}.isCompleted`]: false,
    });
    
    const { path, state } = checkNavigationHelper("info", phase);
    if (task == "info"){
      navigate('/loading', { state: { task: "info", phase: phase } });
    }
    else{
      navigate(userDocData.validation_pending === 0 ? "/results" : path, { state });
    }
  };
  
  const updateUserDoc = async (userDoc, updates) => {
    await updateDoc(userDoc, updates);
  };

  const handleAnswer = async (answer) => { // Added async here
    let correctMappedAnswer;
    const correctAnswer = storedQuestion.correct_answer;
    try {
      correctMappedAnswer = MAPPING_DICT[task][correctAnswer]
    } catch (e) {
      correctMappedAnswer = MAPPING_DICT[task][100]
    }

    if (phase === "train") {
      if (correctMappedAnswer === answer) { // if answer is correct 
        let checkToShowTraining = await checkIfContinueTraining(); // Added await here
        if (checkToShowTraining) {
          const { path, state } = checkNavigationHelper(task, phase);
          navigate(path, { state });
        } else {
          const { path, state } = checkNavigationHelper("drone-damage", "val");
          alert("You have Completed the Training ! You will now be Scored for your Answers, Good Luck !")
          navigate(path, { state });
        }
      } 
      else {
        // send alert pop up with correct answer and ask user to select the correct answer . 
        alert("Incorrect answer. The correct answer is " + correctMappedAnswer);
        const { path, state } = checkNavigationHelper(task, phase);
        navigate(path, { state });
      }
    }
    else if (phase === "val") {
      handleValidationPhase(storedQuestion,answer,correctMappedAnswer === answer);
    }
  }

  if (loading) {
    return <Loader />;
  }

  return (
    <div>
      <Question
        question={storedQuestion}
        handleAnswer={handleAnswer}
      />
    </div>
  );
};

export default Quiz;
