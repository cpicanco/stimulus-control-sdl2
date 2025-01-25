{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye.tracker.types;

{$mode ObjFPC}{$H+}

interface

uses SDL2, Math;

type

  TEyeTrackerCode = (etNone, etGazepoint, etEyeLink, etPupilLabs);

  TNormalizedGaze = record
    X : Float;
    Y : Float;
  end;

  TNormalizedGazes = array of TNormalizedGaze;

  TGazeOnScreenEvent = procedure(AGazes : TNormalizedGazes) of object;

function NormToScreen(AGaze: TNormalizedGaze; AScreen: TSDL_Rect): TSDL_Point;

const
  EYE_TRACKER_GAZE_EVENT = SDL_USEREVENT+3;

implementation

function NormToScreen(AGaze: TNormalizedGaze; AScreen: TSDL_Rect): TSDL_Point;
begin
  Result.X := Round(AGaze.X * AScreen.w);
  Result.Y := Round(AGaze.Y * AScreen.h);
  //Result.Y := Round((1.0 - AGaze.Y) * AScreen.h); For Pupil Labs
end;


end.

