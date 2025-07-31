{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit generics.aggregator.helpseries;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils
   , Generics.Aggregator.Contract
   , Generics.Aggregator
   , Generics.Iterator.Contract
   , Generics.Iterator;

type

  { THelpSeriesAggregator }

  generic THelpSeriesAggregator<_GT> = class abstract (specialize TAggregator<_GT>, specialize IAggregator<_GT>)
  private

  protected
    function Loaded : Boolean;
    procedure LoadFromFile(AFilename : string); virtual; abstract;
    procedure LoadFromMock; virtual; abstract;
  public
    procedure AssignCurrent(AParameters : TStringList); virtual; abstract;
    procedure AssignParameters(AParameters : TStringList); virtual; abstract;
  end;

implementation

{ THelpSeriesAggregator }

function THelpSeriesAggregator.Loaded: Boolean;
begin
  Result := FList.Count > 0;
end;


end.

