{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.markers;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, Generics.Collections
  , SDL2
  , sdl.app.graphics.marker
  ;

type

  TMarkerList = specialize TObjectList<TMarker>;

  { TMarkers }

  TMarkers = class
    private
      FMarkers : TMarkerList;
      FMonitor : TSDL_Rect;
    public
      procedure Paint;
      constructor Create;
      destructor Destroy; override;
      procedure LoadFromFile;
  end;

var
  Markers : TMarkers;

implementation

uses sdl.app.video.methods, session.strutils, session.parameters.global;

{ TMarkers }

constructor TMarkers.Create;
begin
  FMarkers := TMarkerList.Create;
end;

destructor TMarkers.Destroy;
begin
  FMarkers.Free;
  inherited Destroy;
end;

procedure TMarkers.Paint;
var
  LMarker : TMarker;
begin
  for LMarker in FMarkers do begin
    LMarker.Paint;
  end;
end;

procedure TMarkers.LoadFromFile;
var
  LMarkerID: Integer;
  LMarker : TMarker;
  LFilename: String;
begin
  repeat
    FMarkers.Add(TMarker.Create);
  until FMarkers.Count = 4;
  FMonitor := MonitorFromWindow;
  for LMarkerID := 0 to FMarkers.Count -1 do begin
    LFilename := Format('marker_%d', [LMarkerID]);
    LMarker := FMarkers.Items[LMarkerID];
    LMarker.LoadFromFile(AsMarker(LFilename));
    LMarker.Width := GlobalTrialParameters.MarkerSize;
    LMarker.Height := GlobalTrialParameters.MarkerSize;
    LMarker.Parent := FMonitor;
    case LMarkerID of
      0 : LMarker.ToTopLeft;
      1 : LMarker.ToTopRight;
      2 : LMarker.ToBottomLeft;
      3 : LMarker.ToBottomRight;
    end;
  end;
end;

end.

