import React from 'react';
import '../index.css';

function Pricing() {
  return (
    <div className="pricing">
      <h1>Pricing</h1>
      <div className="pricing-tiers">
        <div className="tier">
          <h2>Free Tier</h2>
          <p>Limited features</p>
          <p>$0/month</p>
        </div>
        <div className="tier">
          <h2>Pro Tier</h2>
          <p>Advanced features</p>
          <p>$29/month</p>
        </div>
        <div className="tier">
          <h2>Pro Plus Tier</h2>
          <p>All features</p>
          <p>$99/month</p>
        </div>
      </div>
    </div>
  );
}

export default Pricing;