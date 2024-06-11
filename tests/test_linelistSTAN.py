# tests/test_linelistSTAN.py

import unittest
import pandas as pd
from linelistSTAN import convert_to_linelist, run_model, create_caseCounts

class TestLinelistSTAN(unittest.TestCase):
    def test_convert_to_linelist(self):
        # Create sample data
        case_counts = pd.DataFrame({
            'report_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'onset_date': pd.to_datetime(['2022-12-25', '2022-12-26', '2022-12-27']),
            'delay_int': [7, 7, 7],
            'cases': [10, 15, 20]
        })
        
        # Call the function
        linelist = convert_to_linelist(case_counts)
        
        # Check the output
        self.assertIsInstance(linelist, pd.DataFrame)
        self.assertEqual(len(linelist), 3)
        self.assertIn('value', linelist.columns)
        self.assertIn('id', linelist.columns)
        self.assertIn('start_dt', linelist.columns)
        self.assertIn('actual_onset', linelist.columns)
        self.assertIn('actual_report', linelist.columns)
        self.assertIn('week_int', linelist.columns)
        self.assertIn('is_weekend', linelist.columns)
    
    def test_run_model(self):
        # Create sample data
        data = pd.DataFrame({
            'onset_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'delay_int': [7, 7, 7],
            'actual_report': pd.to_datetime(['2023-01-08', '2023-01-09', '2023-01-10'])
        })
        n_weeks = 2
        
        # Call the function
        predicted_onset = run_model(data, n_weeks)
        
        # Check the output
        self.assertIsInstance(predicted_onset, pd.DataFrame)
        self.assertIn('pred_onset', predicted_onset.columns)
        self.assertIn('nx', predicted_onset.columns)
        self.assertIn('lb', predicted_onset.columns)
        self.assertIn('ex', predicted_onset.columns)
        self.assertIn('ub', predicted_onset.columns)
    
    def test_create_caseCounts(self):
        # Create sample data
        sample_dates = ['2023-01-01', '2023-01-02', '2023-01-03']
        sample_location = ['Location1', 'Location1', 'Location2']
        sample_cases = [10, 15, 20]
        
        # Call the function
        case_counts = create_caseCounts(sample_dates, sample_location, sample_cases)
        
        # Check the output
        self.assertIsInstance(case_counts, pd.DataFrame)
        self.assertEqual(len(case_counts), 3)
        self.assertIn('report_date', case_counts.columns)
        self.assertIn('location', case_counts.columns)
        self.assertIn('cases', case_counts.columns)
        self.assertIn('onset_date', case_counts.columns)
        self.assertIn('delay_int', case_counts.columns)

if __name__ == '__main__':
    unittest.main()