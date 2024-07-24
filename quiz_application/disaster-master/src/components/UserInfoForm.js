import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { doc, updateDoc, getDoc } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';
import { useAuth } from '../context/AuthContext';
import '../styles/UserInfoForm.css';

const UserInfoForm = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const [formData, setFormData] = useState({
    nationality: '',
    age: '',
    gender: '',
    experience: {
      decision_maker: false,
      volunteer: false,
      victim: false
    }
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (type === 'checkbox') {
      setFormData((prevData) => ({
        ...prevData,
        experience: {
          ...prevData.experience,
          [name]: checked
        }
      }));
    } else {
      setFormData((prevData) => ({
        ...prevData,
        [name]: value
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const userDoc = doc(db, 'users', user.uid);
    await updateDoc(userDoc, {
      nationality: formData.nationality,
      age: parseInt(formData.age, 10),
      gender: formData.gender,
      experience: formData.experience
    });
    // navigate('/training/victim/checkaid', { state: { task: 'info', phase: 'train' } });
    navigate('/tutorial/data');
  };

  return (
    <div className="user-info-form-page">
      <div className="form-container">
        <h1>User Information</h1>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="nationality">Nationality:</label>
            <input
              type="text"
              id="nationality"
              name="nationality"
              value={formData.nationality}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="age">Age:</label>
            <input
              type="number"
              id="age"
              name="age"
              value={formData.age}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="gender">Gender:</label>
            <select
              id="gender"
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              required
            >
              <option value="">Select Gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
              <option value="prefer_not_to_say">Prefer not to say</option>
            </select>
          </div>
          <div className="form-group">
            <label>Experience:</label>
            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  name="decision_maker"
                  checked={formData.experience.decision_maker}
                  onChange={handleChange}
                />
                Decision Maker in Disaster Scenarios
              </label>
              <label>
                <input
                  type="checkbox"
                  name="volunteer"
                  checked={formData.experience.volunteer}
                  onChange={handleChange}
                />
                Volunteer in Disaster Scenarios
              </label>
              <label>
                <input
                  type="checkbox"
                  name="victim"
                  checked={formData.experience.victim}
                  onChange={handleChange}
                />
                Victim of a Disaster
              </label>
            </div>
          </div>
          <button type="submit" className="submit-button">Submit</button>
          <p className="disclaimer-text">All data is kept anonymous and the information provided here is used for socio-cultural characteristic comparisons.</p>
        </form>
      </div>
    </div>
  );
};

export default UserInfoForm;
