import React from 'react';
import { useNavigate } from 'react-router-dom';
import useHelperData from '../hooks/useHelperData';
import '../styles/HelpPage.css';

const taskList = ["info", "human", "damage", "satellite", "drone-damage"];
const phase = "train";

const mappingDict = {
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
};

const HelpPage = () => {
  const navigate = useNavigate();

  return (
    <div className="help-page">
      <h1>Survey Task Examples</h1>
      {taskList.map((task) => (
        <TaskSection key={task} task={task} phase={phase} />
      ))}
      <button onClick={() => navigate('/tutorial/game')} className="next-button">
        Next Page
      </button>
    </div>
  );
};

const TaskSection = ({ task, phase }) => {
  const { helperData, loading } = useHelperData(task, phase);

  const taskDescriptions = {
    info: `In this task, you need to determine whether the data (image + text) is informative or not. Informative data relates to humanitarian information about the disaster, while non-informative data does not. Make a judgment based on the data and select your decision.`,
    human: `In this task, you need to determine the type of humanitarian aid related to the data (image + text). The categories are "Affected Individual", "Infrastructure and Utility Damage", "Other Relevant Information", or "Rescue Volunteering or Donation Effort". Make a judgment based on the data and select your decision.`,
    damage: `In this task, you need to determine the extent of damage caused by the disaster as shown in the data (image + text). The categories are "Little or No Damage" or "Severe Damage". Make a judgment based on the data and select your decision.`,
    satellite: `In this task, you need to determine the extent of damage seen from satellite imagery (image only) due to the disaster. The categories are "No Damage" or "Major Damage". Make a judgment based on the data and select your decision.`,
    drone: `In this task, you need to determine the extent of damage seen from surveillance drone imagery (image only) due to the disaster. The categories are "No Damage" or "Damaged". Make a judgment based on the data and select your decision.`,
  };

  if (loading) return <div className="helper-popup">Loading...</div>;

  return (
    <div className="task-section">
      <h2>{task.charAt(0).toUpperCase() + task.slice(1)} Task</h2>
      <p>{taskDescriptions[task]}</p>
      {helperData.map((data) => (
        <div key={data.id} className="helper-item">
          <h3>{mappingDict[task][data.correct_answer]}</h3>
          <p>{data.text}</p>
          {data.image && (
            <div className="helper-image-container">
              <img src={data.image} alt="helper example" className="helper-image" />
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default HelpPage;
