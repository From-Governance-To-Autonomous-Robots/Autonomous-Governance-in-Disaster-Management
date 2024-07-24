import React from 'react';
import useHelperData from '../hooks/useHelperData';
import '../styles/HelperPopup.css';

const HelperPopup = ({ task, phase, mappingDict, onClose }) => {
  const { helperData, loading } = useHelperData(task, phase);

  if (loading) return <div className="helper-popup">Loading...</div>;

  const taskDescriptions = {
    info: `In this task, you need to determine whether the data (image + text) is informative or not. Informative data relates to humanitarian information about the disaster, while non-informative data does not. Make a judgment based on the data and select your decision.`,
    human: `In this task, you need to determine the type of humanitarian aid related to the data (image + text). The categories are "Affected Individual", "Infrastructure and Utility Damage", "Other Relevant Information", or "Rescue Volunteering or Donation Effort". Make a judgment based on the data and select your decision.`,
    damage: `In this task, you need to determine the extent of damage caused by the disaster as shown in the data (image + text). The categories are "Little or No Damage" or "Severe Damage". Make a judgment based on the data and select your decision.`,
    satellite: `In this task, you need to determine the extent of damage seen from satellite imagery (image only) due to the disaster. The categories are "No Damage" or "Major Damage". Make a judgment based on the data and select your decision.`,
    drone: `In this task, you need to determine the extent of damage seen from surveillance drone imagery (image only) due to the disaster. The categories are "No Damage" or "Damaged". Make a judgment based on the data and select your decision.`,
  };

  const additionalDataDescription = `If you are unsure about the data, you can select "Unsure. More Data needed" to request additional data for better judgment. This will provide new data for you to evaluate and make a decision.`;

  return (
    <div className="helper-popup-overlay">
      <div className="helper-popup">
        <button className="close-button" onClick={onClose}>X</button>
        <h2>Helper Information</h2>
        <p>{taskDescriptions[task]}</p>
        <p>{additionalDataDescription}</p>
        {helperData.map((data) => (
          <div key={data.id} className="helper-item">
            <h3>{mappingDict[task][data.correct_answer]}</h3>
            <p>{data.text}</p>
            {data.image && <div className="helper-image-container"><img src={data.image} alt="helper example" className="helper-image" /></div>}
          </div>
        ))}
      </div>
    </div>
  );
};

export default HelperPopup;
