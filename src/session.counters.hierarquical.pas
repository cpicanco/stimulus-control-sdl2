{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.counters.hierarquical;

{$mode ObjFPC}{$H+}
{$modeswitch AdvancedRecords}

interface

uses
  Classes, SysUtils;

type

  { TIDCounter }

  TIDCounter = record
    Count : SmallInt;
    procedure Increment;
  end;

  TIDTrialCounts = array of TIDCounter;

  { TIDBlockCounter }

  TIDBlockCounter = record
    Count : SmallInt;
    Trial : TIDTrialCounts;
    procedure Increment;
  end;

  TIDBlockCounts = array of TIDBlockCounter;

  { TIDSessionCounter }

  TIDSessionCounter = record
    Block : TIDBlockCounts;
    procedure Initialize;
  end;

implementation

uses session.configurationfile;

{ TIDCounter }

procedure TIDCounter.Increment;
begin
  Inc(Count);
end;

{ TIDBlockCounter }

procedure TIDBlockCounter.Increment;
begin
  Inc(Count);
end;

{ TIDSessionCounter }

procedure TIDSessionCounter.Initialize;
var
  i, j: Integer;
begin
  SetLength(Block, ConfigurationFile.TotalBlocks);
  for i := Low(Block) to High(Block) do begin
    Block[i].Count := 0;
    SetLength(Block[i].Trial, ConfigurationFile.Block[i].TotalTrials);
    for j := Low(Block[i].Trial) to High(Block[i].Trial) do begin
      Block[i].Trial[j].Count := 0;
    end;
  end;
end;

end.

