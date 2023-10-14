{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.trials.instruction;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , session.configuration
  , sdl.app.trials
  , sdl.app.stimuli.contract
  , sdl.app.stimuli.instruction
  ;

type

  { TInstruction }

  TInstruction = class sealed (TTrial)
    private
      FStimuli : TInstructionStimuli;
    protected
      function GetIStimuli: IStimuli; override;
      procedure SetTrialData(ATrialData: TTrialData); override;
    public
      constructor Create(AOwner: TComponent); override;
      destructor Destroy; override;
      procedure EndTrial; override;
      procedure Show; override;
      procedure Hide; override;
  end;


implementation

{ TInstruction }

function TInstruction.GetIStimuli: IStimuli;
begin
  Result := FStimuli.AsInterface;
end;

procedure TInstruction.SetTrialData(ATrialData: TTrialData);
begin
  inherited SetTrialData(ATrialData);
end;

constructor TInstruction.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FStimuli := TInstructionStimuli.Create(Self);
  FStimuli.OnFinalize := @EndTrialCallBack;
end;

destructor TInstruction.Destroy;
begin
  { free stuff }
  inherited Destroy;
end;

procedure TInstruction.EndTrial;
begin
  inherited EndTrial;
end;

procedure TInstruction.Show;
begin
  inherited Show;
  //FIStimuli.Start;
end;

procedure TInstruction.Hide;
begin
  inherited Hide;
  //FIStimuli.Stop;
end;

end.

