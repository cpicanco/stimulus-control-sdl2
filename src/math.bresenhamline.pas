{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit math.bresenhamline;

{$mode ObjFPC}{$H+}

interface

uses Classes;

type
  TPoints = array of TPoint;

function BresenhamLine(x0, x1, y0, y1 : integer): TPoints;

implementation

function BresenhamLine(x0, x1, y0, y1: integer): TPoints;
var
    dx, dy, sx, sy, err, err2 : integer;
begin
  Result := nil;
  SetLength(Result,0);
  dx := abs(x1-x0);
  dy := abs(y1-y0);
  if x0 < x1 then sx := 1 else sx := -1;
  if y0 < y1 then sy := 1 else sy := -1;
  err := dx-dy;

  while True do
    begin
     //Plot(x0,y0);
     SetLength(Result, Length(Result) + 1);
     Result[High(Result)] := Point(x0, y0);
     if (x0 = x1) and (y0 = y1) then Break;

     err2 := 2*err;
     if err2 > -dy then
       begin
         err := err - dy;
         x0 := x0 + sx;
       end;
     if err2 < dx then
       begin
         err := err + dx;
         y0 := y0 + sy;
       end;
  end;
end;

end.

