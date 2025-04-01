// import React from 'react';

// import '../index.css';

// function About() {
//   return (
//     <div className="about">
//       <h1>About Us</h1>
//       <p>Meet the team behind SnapVision:</p>
      
//       <div className="developer-cards">
//         {/* Developer 1 */}
//         <div className="developer-card">
//           <div className="developer-image">
//             <img 
//               src="./src/assets/sushrut.jpeg" 
//               alt="Sushrut Lachure" 
//               style={{ width: '100px', height: '100px', objectFit: 'cover' }} 
//             />
//           </div>
//           <a href="https://www.linkedin.com/in/sushrut-lachure/" target="_blank" rel="noopener noreferrer">
//             <h3>Sushrut Lachure</h3>
//           </a>
//           <p>Developer</p>
//         </div>

//         {/* Developer 2 */}
//         <div className="developer-card">
//           <div className="developer-image">
//             <img 
//               src="./src/assets/krisht.jpeg" 
//               alt="Sushrut Lachure" 
//               style={{ width: '100px', height: '100px', objectFit: 'cover' }}
//             />
//           </div>
//           <a href="https://www.linkedin.com/in/krishiiitp/" target="_blank" rel="noopener noreferrer">
//             <h3>Krish Thakrar</h3>
//           </a>
//           <p>Developer</p>
//         </div>

//         {/* Developer 3 */}
//         <div className="developer-card">
//           <div className="developer-image">
//             <img 
//               src="./src/assets/krishy.jpeg" 
//               alt="Sushrut Lachure" 
//               style={{ width: '100px', height: '100px', objectFit: 'cover' }}
//             />
//           </div>
          
//           <a href="https://www.linkedin.com/in/krish-yadav-link17/" target="_blank" rel="noopener noreferrer">
//           <h3>Krish Yadav</h3>  
//           </a>
//           <p>Developer</p>
//         </div>

//         {/* Developer 4 */}
//         <div className="developer-card">
//           <div className="developer-image">
//             <img 
//               src="./src/assets/yash.jpeg" 
//               alt="Sushrut Lachure" 
//               style={{ width: '100px', height: '100px', objectFit: 'cover' }}
//             />
//           </div>
          
//           <a href="https://www.linkedin.com/in/yash-harne/" target="_blank" rel="noopener noreferrer">
//           <h3>Yash Harne</h3>
//           </a>
//           <p>Developer</p>
//         </div>

//         {/* Developer 5 */}
//         <div className="developer-card">
//           <div className="developer-image">
//             <img 
//               src="./src/assets/dhruva.jpeg" 
//               alt="Sushrut Lachure" 
//               style={{ width: '100px', height: '100px', objectFit: 'cover' }}
//             />
//           </div>
          
//           <a href="https://www.linkedin.com/in/dhruva-upadhyaya-94681726b/" target="_blank" rel="noopener noreferrer">
//           <h3>Dhurva Upadhyaya</h3>
//           </a>
//           <p>Developer</p>
//         </div>
//       </div>
//     </div>
//   );
// }

// export default About;


import React from 'react';
import '../index.css';

function About() {
  return (
    <div className="about">
      <h1>About Us</h1>
      <p className="about-description">Meet the team behind SnapVision:</p>
      
      <div className="developer-cards">
        {/* Developer 1 */}
        <div className="developer-card">
          <div className="developer-image">
            <img 
              src="./src/assets/sushrut.jpeg" 
              alt="Sushrut Lachure" 
              style={{ width: '100px', height: '100px', objectFit: 'cover' }} 
            />
          </div>
          <a href="https://www.linkedin.com/in/sushrut-lachure/" target="_blank" rel="noopener noreferrer">
            <h3>Sushrut Lachure</h3>
          </a>
          <p>Developer</p>
        </div>

        {/* Developer 2 */}
        <div className="developer-card">
          <div className="developer-image">
            <img 
              src="./src/assets/krisht.jpeg" 
              alt="Krish Thakrar" 
              style={{ width: '100px', height: '100px', objectFit: 'cover' }}
            />
          </div>
          <a href="https://www.linkedin.com/in/krishiiitp/" target="_blank" rel="noopener noreferrer">
            <h3>Krish Thakrar</h3>
          </a>
          <p>Developer</p>
        </div>

        {/* Developer 3 */}
        <div className="developer-card">
          <div className="developer-image">
            <img 
              src="./src/assets/krishy.jpeg" 
              alt="Krish Yadav" 
              style={{ width: '100px', height: '100px', objectFit: 'cover' }}
            />
          </div>
          <a href="https://www.linkedin.com/in/krish-yadav-link17/" target="_blank" rel="noopener noreferrer">
            <h3>Krish Yadav</h3>
          </a>
          <p>Developer</p>
        </div>

        {/* Developer 4 */}
        <div className="developer-card">
          <div className="developer-image">
            <img 
              src="./src/assets/yash.jpeg" 
              alt="Yash Harne" 
              style={{ width: '100px', height: '100px', objectFit: 'cover' }}
            />
          </div>
          <a href="https://www.linkedin.com/in/yash-harne/" target="_blank" rel="noopener noreferrer">
            <h3>Yash Harne</h3>
          </a>
          <p>Developer</p>
        </div>

        {/* Developer 5 */}
        <div className="developer-card">
          <div className="developer-image">
            <img 
              src="./src/assets/dhruva.jpeg" 
              alt="Dhruva Upadhyaya" 
              style={{ width: '100px', height: '100px', objectFit: 'cover' }}
            />
          </div>
          <a href="https://www.linkedin.com/in/dhruva-upadhyaya-94681726b/" target="_blank" rel="noopener noreferrer">
            <h3>Dhruva Upadhyaya</h3>
          </a>
          <p>Developer</p>
        </div>
      </div>
    </div>
  );
}

export default About;