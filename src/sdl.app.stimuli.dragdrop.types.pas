{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.dragdrop.types;

{$mode ObjFPC}{$H+}

{$modeswitch TypeHelpers}

interface

uses Classes, SysUtils;

type

  TDragDropOrientation =
    (None, TopToBottom, BottomToTop, LeftToRight, RightToLeft, Random);

  TDragDropOrientationRange =
    TDragDropOrientation.None..TDragDropOrientation.Random;

  { TDragDropOrientationHelper }

  TDragDropOrientationHelper = type helper for TDragDropOrientation
    function ToString : string;
  end;

  { THelpSeriesStringHelper }

  THelpSeriesStringHelper = type helper(TStringHelper) for string
    function ToDragDropOrientation : TDragDropOrientation;
  end;

implementation

{ TDragDropOrientationHelper }

function TDragDropOrientationHelper.ToString: string;
begin
  WriteStr(Result, Self);
end;

{ THelpSeriesStringHelper }

function THelpSeriesStringHelper.ToDragDropOrientation: TDragDropOrientation;
var
  LDragDropOrientation : string = '';
begin
  for Result in TDragDropOrientationRange do begin
    WriteStr(LDragDropOrientation, Result);
    if LDragDropOrientation=Self then begin
      Exit;
    end;
  end;
  raise Exception.CreateFmt('TDragDropOrientation %s not found', [Self]);
end;

end.

