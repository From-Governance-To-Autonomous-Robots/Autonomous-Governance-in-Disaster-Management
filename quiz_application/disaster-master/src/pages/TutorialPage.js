import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/TutorialPage.css';
import slide1Image from '../assets/decision_tree.png';
import slide2Image from '../assets/slide_train.png';
import slide3Image from '../assets/slide_help.png';
import slide4Image from '../assets/slide_score.png';
import slide5Image from '../assets/slide_addition.png';
import slide6Image from '../assets/slide_end.png';

const slides = [
  {
    image: slide1Image,
    text: `The user is shown data points relating to the disaster and is asked to make a decision. 1 Scenario is completed when the user has made a decision for all 5 levels in the scenario which involve: 

    Informative vs Not informative data
    Type of Informative data
    Type of damage
    Post-disaster damage assessment using satellite data
    Post-disaster damage assessment using drone data`
  },
  {
    image: slide2Image,
    text: `The user will be shown a Training to get a Feel of how to participate in the Game. Once the Training is Completed you will see a Popup alert saying "You have Completed the Training! You will now be Scored for your Answers, Good Luck!" Again you will start getting scored.`
  },
  {
    image: slide3Image,
    text: `You will have access to a Help Button during the Game, in case you want to see some training examples of what types of data points are seen in that level. This can help aid the decision you make in this level.`
  },
  {
    image: slide4Image,
    text: `You will see Overall Score and Scenario Score. Scenario Score represents the score for the decisions made in that scenario. And Overall score is the aggregate score for all the scenarios the user has completed. You will be given a score of +1 for correct decisions in each level, -1 for requesting additional data, and -5 for every wrong decision.`
  },
  {
    image: slide5Image,
    text: `You will see a button "Unsure. More data needed". Use this if you cannot make a decision on the data you are seeing. For each level in 1 scenario you have 5 chances to request additional data, however each chance can have a -1 to your score.`
  },
  {
    image: slide6Image,
    text: `Click on End Game button, after you have completed 5 or more Scenarios.`
  }
];

const TutorialPage = () => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const navigate = useNavigate();

  const nextSlide = () => {
    if (currentSlide < slides.length - 1) {
      setCurrentSlide(currentSlide + 1);
    }
  };

  const prevSlide = () => {
    if (currentSlide > 0) {
      setCurrentSlide(currentSlide - 1);
    }
  };

  return (
    <div className="tutorial-page">
      <div className="slide">
        <div className="slide-image-container">
          <img src={slides[currentSlide].image} alt={`Slide ${currentSlide + 1}`} className="slide-image" />
        </div>
        <p className="slide-text">{slides[currentSlide].text}</p>
      </div>
      <div className="navigation-buttons">
        <button onClick={prevSlide} disabled={currentSlide === 0} className="nav-button">Previous</button>
        {currentSlide === slides.length - 1 ? (
          <button onClick={() => navigate('/training/victim/checkaid', { state: { task: 'info', phase: 'train' } })} className="nav-button">
            Start Training
          </button>
        ) : (
          <button onClick={nextSlide} className="nav-button">Next</button>
        )}
      </div>
    </div>
  );
};

export default TutorialPage;
