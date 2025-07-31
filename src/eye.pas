{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye;

interface

function VisualAngle(Value, Distance: Double): Double;

implementation

{ Monitor size }

const
  Width = 41.476 { cm };
  Height = 25.922 { cm };
  LetterWidth = 2.6 { cm };
  LetterHeight = 5.3 { cm };
  WordWidth = 13.7 { cm };

uses
  Math;

function RadToDeg(Rad: Double): Double;
begin
  Result := Rad * (180 / PI);
end;

function VisualAngle(Value, Distance: Double): Double;
begin
  Result := 2 * RadToDeg(ArcTan(Value / (2 * Distance)));
end;

end.
