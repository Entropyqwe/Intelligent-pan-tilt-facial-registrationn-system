"""Exercise real Output thread, CSV and invalid exit packet with serial loopback."""
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from core import Snapshot, Observation
from main import Output
from steering import Steering

class OutputSteeringTests(unittest.TestCase):
    def test_loopback_heartbeat_and_exit_neutral(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            cfg=json.loads((Path(__file__).resolve().parents[1]/'steering_config.json').read_text())
            cfg['horizontal_fov_deg']=60
            path=root/'cfg.json'; path.write_text(json.dumps(cfg))
            s=Steering(path)
            log=root/'run.csv'
            output=Output('loop://',log,.4,120,steering=s)
            now=time.monotonic()
            snap=Snapshot(1,True,now,Observation(1,440,80,80,300,.9,.8),640,480,2,now)
            try:
                output.publish(snap)
                time.sleep(.25)
                self.assertIsNone(output.error)
            finally:
                output.close()
            with log.open(encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            self.assertTrue(any(r['valid']=='1' and int(r['bearing_mdeg'])>0 and
                                int(r['servo_us'])>1500 and r['source']=='2' for r in rows))
            self.assertEqual(rows[-1]['valid'],'0')
            self.assertEqual(rows[-1]['servo_us'],'1500')
            self.assertEqual(rows[-1]['source'],'0')

if __name__=='__main__': unittest.main()
