import altrios as alt
import altrios as alt
try:
    print(f"SpeedLimitTrainSim methods: {dir(alt.SpeedLimitTrainSim)}")
    if hasattr(alt, 'TimedLinkPath'):
         print(f"TimedLinkPath methods: {dir(alt.TimedLinkPath)}")
    else:
         print("TimedLinkPath not found")
except Exception as e:
    print(f"Error: {e}")