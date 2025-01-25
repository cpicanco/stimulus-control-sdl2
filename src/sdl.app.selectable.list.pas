{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.selectable.list;

{$mode ObjFPC}{$H+}

interface

uses Generics.Collections, Generics.Defaults, sdl.app.selectable.contract;

type

  { TSelectableByOriginComparer }

  TSelectableByOriginComparer = class (specialize TComparer<ISelectable>)
    function Compare(const AItem1, AItem2: ISelectable): Integer; override;
  end;

  { TSelectables }

  TSelectables = class(specialize TList<ISelectable>)
    function ToJSON : string;
    class function ByOrigin : specialize IComparer<ISelectable>;
  end;

implementation

uses SysUtils, Math, SDL2, sdl.helpers;

{ TSelectableComparer }

function TSelectableByOriginComparer.Compare(
  const AItem1, AItem2: ISelectable): Integer;
var
  A : TSDL_Point;
  B : TSDL_Point;
begin
  A := AItem1.Origen;
  B := AItem2.Origen;

  if A.x = B.x then begin
    Result := CompareValue(A.y, B.y);
  end else begin
    Result := CompareValue(A.x, B.x);
  end;
end;

function TSelectables.ToJSON: string;
var
  LSeparator : string;
  i: Integer;
begin
  Result := '{}';
  if Count = 0 then Exit;

  Result := '';
  for i := 0 to Count-1 do begin
    if i = 0 then begin
      LSeparator := '';
    end else begin
      LSeparator := ',';
    end;

    with Items[i] do begin
      Result := String.Join(
        LSeparator, [Result,
          CustomName + ':' + GetBoundsRect.ToJSON]);
    end;
  end;
  Result := '{'+Result+'}';
end;

class function TSelectables.ByOrigin : specialize IComparer<ISelectable>;
begin
  // TSelectableByOriginComparer is not reference counted
  Result := TSelectableByOriginComparer.Default;
end;

end.

