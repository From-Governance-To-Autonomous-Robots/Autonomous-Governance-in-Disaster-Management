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
    text: (
      <>
        The user is shown data points relating to the disaster and is asked to make a decision. One scenario is completed when the user has made a decision for all 5 levels in the scenario, which involve:
        <ul>
          <li><strong>Informative vs Not informative data</strong></li>
          <li><strong>Type of Informative data</strong></li>
          <li><strong>Type of damage</strong></li>
          <li><strong>Post-disaster damage assessment using satellite data</strong></li>
          <li><strong>Post-disaster damage assessment using drone data</strong></li>
        </ul>
      </>
    )
  },
  {
    image: slide2Image,
    text: (
      <>
        The user will be shown a <strong>Training</strong> to get a feel of how to participate in the game. Once the training is completed, you will see a popup alert saying <strong style={{ color: '#f39c12' }}>"You have completed the training! You will now be scored for your answers. Good luck!"</strong> Again you will start getting scored.
      </>
    )
  },
  {
    image: slide3Image,
    text: (
      <>
        You will have access to a <strong>Help Button</strong> during the game, in case you want to see some training examples of what types of data points are seen in that level. This can help aid the decision you make in this level.
      </>
    )
  },
  {
    image: slide4Image,
    text: (
      <>
        You will see <strong>Overall Score</strong> and <strong>Scenario Score</strong>. Scenario Score represents the score for the decisions made in that scenario. And Overall Score is the aggregate score for all the scenarios the user has completed. You will be given a score of:
        <ul>
          <li><strong style={{ color: 'green' }}>+1</strong> for correct decisions in each level</li>
          <li><strong style={{ color: 'orange' }}>-1</strong> for requesting additional data</li>
          <li><strong style={{ color: 'red' }}>-5</strong> for every wrong decision</li>
        </ul>
      </>
    )
  },
  {
    image: slide5Image,
    text: (
      <>
        You will see a button <strong>"Unsure. More data needed"</strong>. Use this if you cannot make a decision on the data you are seeing. For each level in one scenario, you have <strong>5 chances</strong> to request additional data; however, each chance can have a <strong style={{ color: 'orange' }}>-1</strong> to your score.
      </>
    )
  },
  {
    image: slide6Image,
    text: (
      <>
        Click on the <strong>End Game button</strong>, after you have completed <strong>5 or more Scenarios</strong>.
      </>
    )
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
        <div className="slide-text">{slides[currentSlide].text}</div>
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
