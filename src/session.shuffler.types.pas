{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.shuffler.types;

{$mode ObjFPC}{$H+}
{$modeswitch AdvancedRecords}

interface

uses Generics.Collections;

type

  TIntArray = array of integer;

  TPositions = array of TIntArray;

  { TItem }

  TItem = record
    ID : integer;
    ReferenceName : string;
    class operator = (A, B: TItem): Boolean;
  end;

  TReferenceList = specialize TList<TItem>;

implementation

{ TItem }

class operator TItem. = (A, B: TItem): Boolean;
begin
  Result := A.ReferenceName = B.ReferenceName;
end;

end.

