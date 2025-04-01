import React from 'react';
import { ClipLoader } from 'react-spinners';

function LoadingSpinner() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <ClipLoader color="#36D7B7" loading={true} size={50} />
    </div>
  );
}

export default LoadingSpinner;