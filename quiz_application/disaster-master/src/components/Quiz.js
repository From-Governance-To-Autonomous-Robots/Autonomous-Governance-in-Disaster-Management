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
    0: "Informative",
    1: "Not-Informative",
    100: "Gather Additional Data"
  },
  "human": {
    0: "Affected Individuals",
    1: "Infrastructure and Utility Damage",
    2: "Other Relevant Information",
    3: "Rescue Volunteering or Donation Effort",
    100: "Gather Additional Data"
  },
  "damage": {
    0: "Little or No Damage",
    1: "Severe Damage",
    100: "Gather Additional Data"
  },
  "satellite": {
    0: "No Damage",
    1: "Major Damage",
    100: "Gather Additional Data"
  },
  "drone-damage": {
    0: "No Damage",
    1: "Damaged",
    100: "Gather Additional Data"
  }
}

const calculateMean= (booleanArray)=>{
  // Convert boolean array to integer array (true -> 1, false -> 0)
  const integerArray = booleanArray.map(value => value ? 1 : 0);

  // Calculate the sum of the array
  const sum = integerArray.reduce((accumulator, currentValue) => accumulator + currentValue, 0);

  // Calculate the mean
  const mean = sum / integerArray.length;

  return mean;
}

const Quiz = ({ task, phase }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [userData, setUserData] = useState(null);
  const [currentTreeLevel, setCurrentTreeLevel] = useState(null);
  const [availableTraining, setAvailableTraining] = useState(null);
  const tasksList = ["info", "human", "damage", "satellite", "drone-damage"]
  const phaseList = ["train", "val"]
  const { storedQuestion, loading } = useFirestore(task, phase);
  const [currentTreeScore,setCurrentTreeScore] = useState(0);
  const [overallScore,setOverallScore] = useState(0);
  const [score, setScore] = useState(0);

  const fetchUserData = async () => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();
    setAvailableTraining(userDocData.training_available)
    setUserData(userDocData);
    setCurrentTreeLevel(userDocData.current_tree_level);

    // Check if game has ended
    if (userDocData.gameEnded) {
      navigate("/results");
      return;
    }

    // calculate score
    // Calculate the score here
    const currentTreeLevel = userDocData.current_tree_level;
    const responses = userDocData.responses;

    let isCorrectlyAnsweredList = [];

    for (let i = 0; i <= currentTreeLevel; i++) {
        const treeKey = `tree_${i}`;
        if (responses.hasOwnProperty(treeKey) && responses[treeKey].hasOwnProperty('isCorrectlyAnswered')) {
            isCorrectlyAnsweredList.push(responses[treeKey].isCorrectlyAnswered);
        }
    }

    const score = isCorrectlyAnsweredList.reduce((acc, val) => acc + val, 0) / isCorrectlyAnsweredList.length;
    setOverallScore(score);
    
    setCurrentTreeScore(responses[`tree_${currentTreeLevel}`].isCorrectlyAnswered)
  };

  useEffect(() => {
    fetchUserData();
  }, [task,phase]);

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

  const handleEndGame = async () => {
    // End game here
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();

    // Update gameEnded to true
    await updateDoc(userDoc, {
        gameEnded: true,
    });

    // Calculate the score here
    const currentTreeLevel = userDocData.current_tree_level;
    const responses = userDocData.responses;

    let isCorrectlyAnsweredList = [];
    let isWronglyAnsweredList = [];
    let isGatherAdditionalDataRequestedList = [];

    for (let i = 0; i <= currentTreeLevel; i++) {
        const treeKey = `tree_${i}`;
        if (responses.hasOwnProperty(treeKey) && responses[treeKey].hasOwnProperty('isCorrectlyAnswered')) {
            isCorrectlyAnsweredList.push(responses[treeKey].isCorrectlyAnswered);
        }
        if (responses.hasOwnProperty(treeKey) && responses[treeKey].hasOwnProperty('isWronglyAnswered')) {
          isWronglyAnsweredList.push(responses[treeKey].isWronglyAnswered);
        }
        if (responses.hasOwnProperty(treeKey) && responses[treeKey].hasOwnProperty('isGatherAdditionalDataRequested')) {
          isGatherAdditionalDataRequestedList.push(responses[treeKey].isGatherAdditionalDataRequested);
        }
    }

    const score = isCorrectlyAnsweredList.reduce((acc, val) => acc + val, 0) / isCorrectlyAnsweredList.length;
    const additionalDataScore = isGatherAdditionalDataRequestedList.reduce((acc, val) => acc + val, 0) / isGatherAdditionalDataRequestedList.length;
    const wronglyAnsweredScore = isWronglyAnsweredList.reduce((acc, val) => acc + val, 0) / isWronglyAnsweredList.length;
    // Update the score in the document
    await updateDoc(userDoc, {
        score: score,
        CorrectlyAnswered:score,
        WronglyAnswered:wronglyAnsweredScore,
        GatherAdditionalDataRequested:additionalDataScore,
    });

    // Navigate to results page
    navigate("/results");
    return;
}

  const handleValidationPhase = async (question, userAnswer, isCorrect) => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();
    
    if (userDocData.gameEnded) {
      navigate("/results");
      return;
    }
  
    let currentTree = userDocData.responses[`tree_${userDocData.current_tree_level}`];
  
    if (isCorrect) {
      handleCorrectAnswer(userDoc, userDocData, question, userAnswer, currentTree);
    } else if (userAnswer === "Gather Additional Data") {
      handleGatherAdditionalData(userDoc, userDocData, question, userAnswer, currentTree);
    } else { 
      handleNoResponseNecessary(userDoc, userDocData, question, userAnswer, currentTree);
    }
    
  };
  
  
  const handleCorrectAnswer = async (userDoc, userDocData, question, userAnswer, currentTree) => {
    
    if (userDocData.responses[`tree_${userDocData.current_tree_level}`].tree_done === false){
      if (task === "drone-damage") {
        await updateUserDoc(userDoc, {
          current_tree_level: userDocData.current_tree_level + 1,
          number_of_completed_trees: userDocData.number_of_completed_trees + 1,
          [`responses.tree_${userDocData.current_tree_level}.tree_done`]: true,
        });
      }
    
      currentTree.points.push(1);
      let index = tasksList.indexOf(task);
      currentTree.isCompleted[index]=true;
      
      currentTree.tree[task].question_id.push(question.question_id);
      currentTree.tree[task].user_answer.push(userAnswer);
    
      const updates = {
        [`responses.tree_${userDocData.current_tree_level}.tree.${task}.question_id`]: currentTree.tree[task].question_id,
        [`responses.tree_${userDocData.current_tree_level}.tree.${task}.user_answer`]: currentTree.tree[task].user_answer,
        [`responses.tree_${userDocData.current_tree_level}.points`]: currentTree.points,
        [`responses.tree_${userDocData.current_tree_level}.isCompleted`]: currentTree.isCompleted,
        [`responses.tree_${userDocData.current_tree_level}.isCorrectlyAnswered`]: calculateMean(currentTree.isCompleted),
      };
      
      await updateUserDoc(userDoc, updates);
    }

    const { path, state }  = checkNavigationHelper(task, phase);
    // navigate(userDocData.validation_pending === 0 ? "/results" : path, { state });
    navigate(path, { state });
  };
  
  const handleGatherAdditionalData = async (userDoc, userDocData, question, userAnswer, currentTree) => {
    if (currentTree.tree[task].availableAdditionalData === 0) {
      alert("You have requested the maximum amount of additional examples. Please select an answer for the task.");
      return;
    }
  
    currentTree.points.push(-1);
    currentTree.tree[task].availableAdditionalData -= 1;
    let index = tasksList.indexOf(task);
    currentTree.gotGatherAdditionalDataRequested[index]=true;
  
    await updateUserDoc(userDoc, {
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.question_id`]: currentTree.tree[task].question_id,
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.user_answer`]: currentTree.tree[task].user_answer,
      [`responses.tree_${userDocData.current_tree_level}.points`]: currentTree.points,
      [`responses.tree_${userDocData.current_tree_level}.gotGatherAdditionalDataRequested`]: currentTree.gotGatherAdditionalDataRequested,
      [`responses.tree_${userDocData.current_tree_level}.isGatherAdditionalDataRequested`]: calculateMean(currentTree.gotGatherAdditionalDataRequested),
      [`responses.tree_${userDocData.current_tree_level}.tree.${task}.availableAdditionalData`]: currentTree.tree[task].availableAdditionalData,
    });
    
    let previousTaskIndex = tasksList.indexOf(task) - 1;
    if (previousTaskIndex <0){
      previousTaskIndex = 4
    }
    let newTask = tasksList[previousTaskIndex]
    const { path, state } = checkNavigationHelper(newTask, phase);
    navigate('/loading', { state: { task: newTask, phase: phase } });
  };

  const handleGatherAdditionalDataTrain = async () => {
    let previousTaskIndex = tasksList.indexOf(task) - 1;
    if (previousTaskIndex <0){
      previousTaskIndex = 4
    }
    let newTask = tasksList[previousTaskIndex]
    const { path, state } = checkNavigationHelper(newTask, phase);
    navigate('/loading', { state: { task: newTask, phase: phase } });
  };
  
  const handleNoResponseNecessary = async (userDoc, userDocData, question, userAnswer, currentTree) => {
    if (userDocData.responses[`tree_${userDocData.current_tree_level}`].tree_done === false){
      if (task === "drone-damage") {
        await updateUserDoc(userDoc, {
          current_tree_level: userDocData.current_tree_level + 1,
          number_of_failed_trees: userDocData.number_of_failed_trees + 1,
          [`responses.tree_${userDocData.current_tree_level}.tree_done`]: true,
        });
      }
    
      currentTree.points.push(-5);
      currentTree.isCompleted[tasksList.indexOf(task)]=false;
      currentTree.gotWronglyAnswered[tasksList.indexOf(task)]=true;
      currentTree.tree[task].question_id.push(question.question_id);
      currentTree.tree[task].user_answer.push(userAnswer);
    
      await updateUserDoc(userDoc, {
        [`responses.tree_${userDocData.current_tree_level}.tree.${task}.question_id`]: currentTree.tree[task].question_id,
        [`responses.tree_${userDocData.current_tree_level}.tree.${task}.user_answer`]: currentTree.tree[task].user_answer,
        [`responses.tree_${userDocData.current_tree_level}.points`]: currentTree.points,
        [`responses.tree_${userDocData.current_tree_level}.isCompleted`]: currentTree.isCompleted,
        [`responses.tree_${userDocData.current_tree_level}.isCorrectlyAnswered`]: calculateMean(currentTree.isCompleted),
        [`responses.tree_${userDocData.current_tree_level}.gotWronglyAnswered`]: currentTree.gotWronglyAnswered,
        [`responses.tree_${userDocData.current_tree_level}.isWronglyAnswered`]: calculateMean(currentTree.gotWronglyAnswered),
      });
    }
    
    const { path, state }  = checkNavigationHelper(task, phase);
    // navigate(userDocData.validation_pending === 0 ? "/results" : path, { state });
    navigate( path, { state });
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
      else if (answer === "Gather Additional Data"){
        handleGatherAdditionalDataTrain();
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
        currentTreeLevel={currentTreeLevel}
        handleAnswer={handleAnswer}
        handleEndGame={handleEndGame}
        mappingDict={MAPPING_DICT}
        task={task}
        currentTreeScore={currentTreeScore}
        overallScore={overallScore}
      />
    </div>
  );
};

export default Quiz;
